# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import shutil
import tempfile
import warnings
import zipfile
from typing import Optional, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; must contain 'model', 'optimizer', 'extra'.
        lora_adapter_only (bool, optional):
            LoRA runs only: save just the adapter parameters in the model shard;
            the frozen base is rebuilt from the HF init before loading. Defaults
            to False; an extra-state marker keeps either layout loadable.
    """

    #: key fragment identifying PEFT/LoRA parameters in the FSDP state dict
    _LORA_KEY_FRAGMENT = "lora_"
    #: marker persisted in extra_state so load can tell which layout it has
    _ADAPTER_ONLY_MARKER = "lora_adapter_only"

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        lora_adapter_only: bool = False,
        base_model_path: Optional[str] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        self.lora_adapter_only = bool(lora_adapter_only)
        self.base_model_path = base_model_path
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def _check_adapter_base_matches(self, local_path: str):
        """Refuse to load an adapter-only checkpoint onto a different base model.

        The base recorded in `lora_adapter/adapter_config.json` must match
        `base_model_path`; skipped when either is unknown.
        """
        if not self.base_model_path:
            return
        cfg_path = os.path.join(local_path, "lora_adapter", "adapter_config.json")
        if not os.path.isfile(cfg_path):
            if self.rank == 0:
                print(f"[checkpoint] No {cfg_path}; cannot verify the adapter's base model.")
            return
        try:
            with open(cfg_path) as f:
                recorded = json.load(f).get("base_model_name_or_path")
        except (OSError, ValueError) as e:
            if self.rank == 0:
                print(f"[checkpoint] Could not read {cfg_path} ({e}); skipping base-model check.")
            return
        if not recorded:
            return
        if os.path.realpath(recorded) != os.path.realpath(self.base_model_path):
            raise RuntimeError(
                f"[rank-{self.rank}]: adapter-only checkpoint was trained on a different base model.\n"
                f"  checkpoint's base : {recorded}\n"
                f"  configured base   : {self.base_model_path}\n"
                "The shard holds no base weights, so resuming would silently train an "
                "adapter on top of the wrong model. Point the run at the original init, "
                "or start a fresh run if the change is intended."
            )

    def _rotate_saved_paths(self, local_path: str, max_ckpt_to_keep) -> list:
        """Record `local_path` as the newest checkpoint and return the stale ones (no filesystem writes).

        A step can be saved twice (best-val save plus the regular save), so any prior entry for
        this path is dropped first and the path just written is never returned for deletion.
        """
        current = os.path.abspath(local_path)
        self.previous_saved_paths = [p for p in self.previous_saved_paths if os.path.abspath(p) != current]
        self.previous_saved_paths.append(local_path)

        if not (max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0):
            return []
        if len(self.previous_saved_paths) <= max_ckpt_to_keep:
            return []

        keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep
        stale = self.previous_saved_paths[:keep_start]
        self.previous_saved_paths = self.previous_saved_paths[keep_start:]
        # Belt and braces: the dedup above should already guarantee this.
        return [p for p in stale if os.path.abspath(p) != current]

    def _validate_checkpoint_file(self, file_path: str) -> bool:
        """
        Validate that a checkpoint file is not corrupted.
        
        Args:
            file_path: Path to the checkpoint file to validate.
            
        Returns:
            True if valid, False if corrupted.
        """
        try:
            # Check if file exists and has non-zero size
            if not os.path.exists(file_path):
                print(f"[rank-{self.rank}]: Checkpoint file does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"[rank-{self.rank}]: Checkpoint file is empty: {file_path}")
                return False
            
            # Try to open as zip (PyTorch saves as zip files)
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Test the zip file integrity
                    bad_file = zf.testzip()
                    if bad_file is not None:
                        print(f"[rank-{self.rank}]: Corrupted file in checkpoint zip: {bad_file}")
                        return False
            except zipfile.BadZipFile:
                print(f"[rank-{self.rank}]: Checkpoint file is not a valid zip: {file_path}")
                return False
            
            return True
        except Exception as e:
            print(f"[rank-{self.rank}]: Error validating checkpoint {file_path}: {e}")
            return False

    def _maybe_stage_local_checkpoint_file(self, file_path: str) -> str:
        """
        Copy a local checkpoint shard to node-local /tmp before torch.load(), which can stall on
        shared network filesystems. Controlled by VERL_STAGE_LOCAL_CKPT_TO_TMP (on/off/auto) and,
        in auto mode, VERL_STAGE_LOCAL_CKPT_PREFIXES (colon-separated path prefixes).
        """
        if not file_path or is_non_local(file_path) or not os.path.isfile(file_path):
            return file_path

        stage_mode = os.environ.get("VERL_STAGE_LOCAL_CKPT_TO_TMP", "auto").strip().lower()
        if stage_mode in {"0", "false", "off", "no"}:
            return file_path

        should_stage = stage_mode in {"1", "true", "on", "yes"}
        if not should_stage:
            should_stage = any(file_path.startswith(p) for p in os.environ.get("VERL_STAGE_LOCAL_CKPT_PREFIXES", "").split(":") if p)

        if not should_stage:
            return file_path

        src_size = os.path.getsize(file_path)
        cache_root = os.path.join(tempfile.gettempdir(), "verl_local_ckpt_cache", hashlib.md5(file_path.encode()).hexdigest())
        os.makedirs(cache_root, exist_ok=True)
        staged_path = os.path.join(cache_root, os.path.basename(file_path))

        if os.path.exists(staged_path) and os.path.getsize(staged_path) == src_size:
            return staged_path

        fd, tmp_path = tempfile.mkstemp(prefix=".partial_", dir=cache_root)
        os.close(fd)
        try:
            shutil.copy2(file_path, tmp_path)
            copied_size = os.path.getsize(tmp_path)
            if copied_size != src_size:
                raise RuntimeError(
                    f"[rank-{self.rank}]: staged checkpoint size mismatch for {file_path}: "
                    f"expected {src_size}, got {copied_size}"
                )
            os.replace(tmp_path, staged_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return staged_path

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)
        local_model_path = self._maybe_stage_local_checkpoint_file(local_model_path)
        local_optim_path = self._maybe_stage_local_checkpoint_file(local_optim_path)
        local_extra_state_path = self._maybe_stage_local_checkpoint_file(local_extra_state_path)

        # Validate checkpoint files before loading
        checkpoint_files = [
            ("model", local_model_path),
            ("optimizer", local_optim_path),
            ("extra_state", local_extra_state_path)
        ]
        
        for name, path in checkpoint_files:
            if not self._validate_checkpoint_file(path):
                raise RuntimeError(
                    f"[rank-{self.rank}]: Corrupted or incomplete {name} checkpoint at {path}. "
                    f"This usually happens when a previous checkpoint save was interrupted. "
                    f"Please delete the corrupted checkpoint directory: {local_path} "
                    f"and restart from an earlier checkpoint."
                )

        model_state_dict = torch.load(local_model_path, weights_only=False)
        optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        # Adapter-only shards hold just the LoRA parameters; the worker has already
        # rebuilt the frozen base, so missing base keys are expected.
        adapter_only = bool(extra_state_dict.get(self._ADAPTER_ONLY_MARKER, False))
        if adapter_only:
            self._check_adapter_base_matches(local_path)

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if adapter_only:
                missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
                # Every saved key must land and no LoRA key may stay unfilled.
                if unexpected:
                    raise RuntimeError(
                        f"[rank-{self.rank}]: adapter-only checkpoint has {len(unexpected)} "
                        f"key(s) the model does not expect, e.g. {list(unexpected)[:3]}"
                    )
                stranded = [k for k in missing if self._LORA_KEY_FRAGMENT in k]
                if stranded:
                    raise RuntimeError(
                        f"[rank-{self.rank}]: adapter-only checkpoint did not restore "
                        f"{len(stranded)} LoRA parameter(s), e.g. {stranded[:3]}. "
                        "Refusing to resume with an untrained adapter."
                    )
                if self.rank == 0:
                    print(
                        f"[checkpoint] Restored {len(model_state_dict)} LoRA keys; "
                        f"{len(missing)} frozen base keys kept from the HF init"
                    )
            else:
                self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def _discover_existing_checkpoints(self, local_path: str):
        """
        Populate previous_saved_paths from existing checkpoint directories on disk.

        Called once on the first save after a resume so that the cleanup logic in
        save_checkpoint() knows about checkpoints written by earlier runs.

        Args:
            local_path: Current save path, e.g. ".../global_step_48/actor".
                        We scan the grandparent for sibling global_step_* dirs.
        """
        # local_path looks like .../global_step_N/<component>
        # component is "actor" or "critic"
        component = os.path.basename(local_path)           # e.g. "actor"
        step_dir = os.path.dirname(local_path)             # e.g. .../global_step_48
        ckpt_root = os.path.dirname(step_dir)              # e.g. .../experiment_name/

        if not os.path.isdir(ckpt_root):
            return

        existing = []
        for entry in os.listdir(ckpt_root):
            if not entry.startswith("global_step_"):
                continue
            try:
                step_num = int(entry.split("global_step_")[1])
            except (ValueError, IndexError):
                continue
            candidate = os.path.join(ckpt_root, entry, component)
            if os.path.isdir(candidate):
                existing.append((step_num, candidate))

        # Sort by step number so oldest is first (cleanup removes from the front)
        existing.sort(key=lambda x: x[0])
        self.previous_saved_paths = [path for _, path in existing]
        if self.previous_saved_paths and self.rank == 0:
            print(
                f"[checkpoint] Discovered {len(self.previous_saved_paths)} existing "
                f"{component} checkpoints for rotation: "
                f"{[os.path.basename(os.path.dirname(p)) for p in self.previous_saved_paths]}"
            )

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # On the first save (e.g. after resume), discover checkpoints already on
        # disk so that the rotation / cleanup logic can account for them.
        if not self.previous_saved_paths and max_ckpt_to_keep:
            self._discover_existing_checkpoints(local_path)

        # record the previous global step
        self.previous_global_step = global_step


        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict() if self.optimizer is not None else None
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                adapter_only = self.lora_adapter_only and any(
                    self._LORA_KEY_FRAGMENT in k for k in model_state_dict
                )
                if self.lora_adapter_only and not adapter_only:
                    # Misconfiguration: fail rather than silently write a full shard.
                    raise RuntimeError(
                        "lora_adapter_only=True but no LoRA parameters were found in the "
                        f"model state dict (rank {self.rank}). Enable LoRA "
                        "(actor_rollout_ref.model.lora_rank>0) or turn the flag off."
                    )
                if adapter_only:
                    full_keys = len(model_state_dict)
                    model_state_dict = {
                        k: v for k, v in model_state_dict.items() if self._LORA_KEY_FRAGMENT in k
                    }
                    if self.rank == 0:
                        print(
                            f"[checkpoint] LoRA adapter-only shard: keeping "
                            f"{len(model_state_dict)}/{full_keys} model keys "
                            "(frozen base is restored from the HF init on load)"
                        )

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                    self._ADAPTER_ONLY_MARKER: adapter_only,
                }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                
                # Use atomic writes: save to temp file then rename
                # This prevents corruption if the process is interrupted during save
                
                # Save model
                model_tmp = model_path + ".tmp"
                torch.save(model_state_dict, model_tmp)
                os.replace(model_tmp, model_path)
                
                # Save optimizer
                optim_tmp = optim_path + ".tmp"
                torch.save(optimizer_state_dict, optim_tmp)  # TODO: address optimizer is None
                os.replace(optim_tmp, optim_path)
                
                # Save extra state
                extra_tmp = extra_path + ".tmp"
                torch.save(extra_state_dict, extra_tmp)
                os.replace(extra_tmp, extra_path)

        if self.rank == 0:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            model_config = unwrap_model.config
            generation_config = getattr(unwrap_model, "generation_config", None)
            if generation_config is None and unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # config.name_or_path may point at a stale checkpoint dir without
                # generation_config.json; saving must not depend on reloading it.
                try:
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                except Exception as e:
                    print(
                        f"Warning: Failed to reload generation config from {model_config.name_or_path}: {e}. "
                        "Continuing without reloading because the in-memory model config is sufficient for checkpoint save."
                    )
                    generation_config = None
            if generation_config is not None:
                generation_config.save_pretrained(local_path)

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if "hf_model" in self.checkpoint_contents:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        # Rotate only after the new checkpoint is fully on disk, so a crash mid-save
        # still leaves a usable checkpoint (transient peak: old + new).
        to_remove = self._rotate_saved_paths(local_path, max_ckpt_to_keep)
        if to_remove:
            self.remove_previous_save_local_path(to_remove)
