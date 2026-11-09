# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
import time
import copy
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from agent_system.environments.prompts.circuitop import CIRCUITOP_SYSTEM_HEADER
from typing import List, Dict, Any, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import os

# Continuation policies (which sampled branch is replayed as the next shared state):
#   greedy     - the branch with the best continuation score
#   softmax    - sample a branch with probability softmax(score / T)
#   random     - uniform over the eligible branches (SLIC-Random, the main method)
#   fixed_base - no online continuation: one base trajectory per group is rolled out
#                first and every turn's candidates branch from its fixed history
#                (the fixed-base baseline)
SLIC_CONTINUATION_POLICIES = ("greedy", "softmax", "random", "fixed_base")
_POLICY_ALIASES = {"best": "greedy", "none": "fixed_base"}


def canonical_continuation_policy(value) -> str:
    policy = str(value).strip().lower()
    return _POLICY_ALIASES.get(policy, policy)

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Optional system message from the env (static context, e.g. circuit topology).
        system_msg = obs.get('system', None)
        
        chat_messages = []
        if system_msg is not None and isinstance(system_msg, (list, np.ndarray)) and len(system_msg) > item:
            system_entry = system_msg[item]
            if system_entry:
                chat_messages.append({
                    "content": f"{CIRCUITOP_SYSTEM_HEADER}{system_entry}",
                    "role": "system",
                })
        
        # User message for dynamic observation
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")
        
        chat_messages.append({
            "content": obs_content,
            "role": "user",
        })
        
        # Keep as list - tokenizer.apply_chat_template expects list, not numpy array
        chat = chat_messages
        
        # CIRCUITOP_DISABLE_THINKING=1 passes enable_thinking=False (Qwen3-style
        # templates default to thinking on); other templates ignore it.
        template_kwargs = {}
        if os.environ.get("CIRCUITOP_DISABLE_THINKING") == "1":
            template_kwargs["enable_thinking"] = False
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
              ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source,
            'formatted_prompt_text': prompt_with_chat_template  # Store original formatted prompt for logging
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat  # Already a list, no need for tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def _is_slic(self) -> bool:
        adv = str(getattr(self.config.algorithm, "adv_estimator", "")).lower()
        return adv == "slic"

    def _is_mt_grpo_tree(self) -> bool:
        # MT-GRPO full-tree baseline: gated on the estimator name alone.
        return str(getattr(self.config.algorithm, "adv_estimator", "")).lower() == "mt_grpo_tree"

    def _slic_cfg_get(self, key: str, default: Any = None) -> Any:
        cfg = getattr(self.config.algorithm, "slic", None)
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
            raise RuntimeError(f"Cannot parse boolean value from string: {value!r}")
        return bool(value)

    def _slic_candidate_score(
        self,
        *,
        info: Dict[str, Any],
        reward: float,
        score_mode: str,
        row_idx: int,
    ) -> float:
        """Return the scalar used only for selecting the next shared SLIC branch."""
        from verl.utils.circuit_step import uses_fixed_failure_reward

        if (self._as_bool(self._slic_cfg_get("use_fixed_invalid_turn_reward", False))
                and uses_fixed_failure_reward(info)):
            return float(self._slic_cfg_get("invalid_turn_reward", -0.1))
        invalid_score = self._slic_cfg_get("continuation_invalid_score", None)
        if invalid_score is not None and int(info.get("is_action_valid", 1)) != 1:
            try:
                return float(invalid_score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"SLIC continuation_invalid_score must be numeric, got {invalid_score!r}."
                ) from exc

        mode = str(score_mode).strip()
        if mode in {"reward", "env_reward", "step_reward"}:
            value = reward
        elif mode == "perf_score_delta":
            if "perf_score_post" not in info or "perf_score_pre" not in info:
                raise RuntimeError(
                    "SLIC continuation score 'perf_score_delta' requires "
                    f"perf_score_post and perf_score_pre in step info at row {row_idx}."
                )
            value = float(info["perf_score_post"]) - float(info["perf_score_pre"])
        else:
            key = mode
            if key.startswith("step_info:"):
                key = key.split(":", 1)[1]
            if key not in info:
                available = sorted(info.keys())
                raise RuntimeError(
                    f"SLIC continuation score key '{key}' is missing at row {row_idx}. "
                    f"Available step_info keys: {available}"
                )
            value = info[key]

        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"SLIC continuation score '{score_mode}' at row {row_idx} is not numeric: {value!r}."
            ) from exc
        if not np.isfinite(score):
            raise RuntimeError(
                f"SLIC continuation score '{score_mode}' at row {row_idx} is not finite: {score!r}."
            )
        return score

    def _slic_candidate_success(self, info: Dict[str, Any]) -> bool:
        keys = self._slic_cfg_get(
            "success_keys",
            ["performance_success", "success"],
        )
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        for key in keys:
            value = info.get(key)
            if isinstance(value, (bool, np.bool_)):
                if bool(value):
                    return True
            elif isinstance(value, (int, float, np.integer, np.floating)):
                if float(value) != 0.0:
                    return True
        return False

    def _select_slic_continuation_indices(
        self,
        *,
        infos: List[Dict[str, Any]],
        rewards: np.ndarray,
        batch_size: int,
        group_size: int,
        group_done: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select one candidate per SLIC group for replay into the next shared state."""
        policy = canonical_continuation_policy(self._slic_cfg_get("continuation_policy", "random"))
        score_mode = str(self._slic_cfg_get("continuation_score", "perf_score_post")).strip()
        valid_only = self._as_bool(self._slic_cfg_get("continuation_valid_only", True))
        # Softmax acts on raw scores, whose within-group spread is small: T=1.0 is near-uniform, lower T sharpens it.
        temperature = float(self._slic_cfg_get("continuation_temperature", 1.0))
        success_priority = self._as_bool(self._slic_cfg_get("success_priority", True))
        terminate_on_any_success = self._as_bool(self._slic_cfg_get("terminate_on_any_success", True))
        terminate_on_any_done = self._as_bool(self._slic_cfg_get("terminate_on_any_done", False))

        if policy not in SLIC_CONTINUATION_POLICIES:
            raise RuntimeError(
                f"Unknown SLIC continuation_policy='{policy}'. "
                "Expected one of: greedy, softmax, random, fixed_base (aliases: best, none)."
            )
        if policy == "softmax" and temperature <= 0.0:
            raise RuntimeError("SLIC continuation_temperature must be > 0 for softmax selection.")

        rewards_np = np.asarray(rewards, dtype=np.float64)
        if rewards_np.shape[0] != batch_size:
            raise RuntimeError(
                f"SLIC rewards size mismatch: got {rewards_np.shape[0]}, expected {batch_size}."
            )

        num_groups = (batch_size + group_size - 1) // group_size
        selected_indices = np.full(num_groups, -1, dtype=np.int64)
        candidate_scores = np.full(batch_size, np.nan, dtype=np.float64)
        all_invalid_fallback = np.zeros(num_groups, dtype=bool)
        group_terminal = np.zeros(num_groups, dtype=bool)

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, batch_size)
            if group_done[g]:
                selected_indices[g] = start
                continue

            group_indices = np.arange(start, end, dtype=np.int64)
            for idx in group_indices:
                candidate_scores[idx] = self._slic_candidate_score(
                    info=infos[int(idx)],
                    reward=float(rewards_np[int(idx)]),
                    score_mode=score_mode,
                    row_idx=int(idx),
                )

            success_mask = np.array(
                [self._slic_candidate_success(infos[int(idx)]) for idx in group_indices],
                dtype=bool,
            )
            done_mask = np.array(
                [bool(infos[int(idx)].get("done", False)) for idx in group_indices],
                dtype=bool,
            )
            terminal_mask = np.zeros_like(success_mask, dtype=bool)
            if terminate_on_any_success:
                terminal_mask |= success_mask
            if terminate_on_any_done:
                terminal_mask |= done_mask
            group_terminal[g] = bool(terminal_mask.any())

            eligible = group_indices
            if success_priority and terminal_mask.any():
                eligible = group_indices[terminal_mask]
            if valid_only:
                valid = np.array(
                    [int(infos[int(idx)].get("is_action_valid", 1)) == 1 for idx in eligible],
                    dtype=bool,
                )
                if valid.any():
                    eligible = eligible[valid]
                elif not (success_priority and terminal_mask.any()):
                    all_invalid_fallback[g] = True

            eligible_scores = candidate_scores[eligible]
            if eligible_scores.size == 0:
                raise RuntimeError(f"SLIC group {g} has no continuation candidates.")

            if policy == "first" and not (success_priority and terminal_mask.any()):
                selected_indices[g] = int(eligible[0])
            elif policy == "greedy" or (success_priority and terminal_mask.any()):
                best_local = int(np.argmax(eligible_scores))
                selected_indices[g] = int(eligible[best_local])
            elif policy == "random":
                # Uniform over the same eligible set (validity / success-priority filters still apply).
                selected_indices[g] = int(np.random.choice(eligible))
            else:
                shifted = (eligible_scores - np.max(eligible_scores)) / temperature
                weights = np.exp(shifted)
                probs = weights / np.sum(weights)
                selected_indices[g] = int(np.random.choice(eligible, p=probs))

        return selected_indices, candidate_scores, all_invalid_fallback, group_terminal

    def _apply_turn_batch_exact_rows(self, data: DataProto) -> DataProto:
        target_rows = int(self._slic_cfg_get("target_train_rows", 0) or 0)
        if target_rows <= 0 or len(data) == target_rows:
            return data

        group_size = int(self.config.env.rollout.n) if int(self.config.env.rollout.n) > 0 else len(data)
        if target_rows % group_size != 0:
            raise RuntimeError(
                f"SLIC target_train_rows={target_rows} must be divisible by group_size={group_size}."
            )
        if "uid" not in data.non_tensor_batch or "turn_index" not in data.non_tensor_batch:
            raise RuntimeError("SLIC exact row selection requires non_tensor_batch['uid'] and ['turn_index'].")

        uid_arr = np.asarray(data.non_tensor_batch["uid"], dtype=object)
        turn_arr = np.asarray(data.non_tensor_batch["turn_index"], dtype=np.int64)
        groups: Dict[tuple[Any, int], List[int]] = {}
        first_pos: Dict[tuple[Any, int], int] = {}
        for row_idx, (uid, turn) in enumerate(zip(uid_arr, turn_arr)):
            key = (uid, int(turn))
            groups.setdefault(key, []).append(row_idx)
            first_pos.setdefault(key, row_idx)

        complete_groups = [
            (key, rows)
            for key, rows in groups.items()
            if len(rows) == group_size
        ]
        complete_groups.sort(key=lambda item: (item[0][1], first_pos[item[0]]))
        if not complete_groups:
            raise RuntimeError("SLIC exact row selection found no complete turn groups.")
        # With `earliest` and target_rows < generated rows, the late turns are never trained on.
        # `turn_stratified` takes one complete group per turn index in round-robin order until the target is met,
        # so every turn depth is represented in proportion to what the rollout produced.
        select_mode = str(self._slic_cfg_get("exact_rows_select", "earliest")).strip().lower()
        if select_mode == "turn_stratified":
            by_turn: Dict[int, List[List[int]]] = {}
            for key, rows in complete_groups:
                by_turn.setdefault(int(key[1]), []).append(rows)
            ordered: List[List[int]] = []
            queues = [by_turn[t] for t in sorted(by_turn)]
            while any(queues):
                for q in queues:
                    if q:
                        ordered.append(q.pop(0))
            complete_groups = [((None, -1), rows) for rows in ordered]
        elif select_mode != "earliest":
            raise RuntimeError(f"Unknown algorithm.step_wise.exact_rows_select={select_mode!r} (earliest | turn_stratified)")

        chunks: List[tuple[List[int], int | None]] = []
        for _, rows in complete_groups:
            if len(chunks) * group_size >= target_rows:
                break
            chunks.append((rows, None))

        fill_mode = str(self._slic_cfg_get("exact_rows_fill_mode", "none")).strip().lower()
        duplicate_id = 0
        if len(chunks) * group_size < target_rows:
            if fill_mode != "duplicate":
                return data.select_idxs(np.asarray([idx for rows, _ in chunks for idx in rows], dtype=np.int64))
            base_chunks = list(chunks)
            if not base_chunks:
                raise RuntimeError("SLIC exact row duplication requires at least one selected complete group.")
            while len(chunks) * group_size < target_rows:
                rows = base_chunks[duplicate_id % len(base_chunks)][0]
                chunks.append((rows, duplicate_id))
                duplicate_id += 1

        selected_indices = np.asarray([idx for rows, _ in chunks for idx in rows], dtype=np.int64)
        out = data.select_idxs(selected_indices)

        # Break duplicated turn groups into distinct GRPO groups, otherwise the
        # advantage baseline would merge originals and duplicates.
        if any(dup_id is not None for _, dup_id in chunks):
            for key, value in list(out.non_tensor_batch.items()):
                if isinstance(value, np.ndarray) and value.dtype == object:
                    out.non_tensor_batch[key] = np.asarray([copy.deepcopy(v) for v in value], dtype=object)

            uid_out = np.asarray(out.non_tensor_batch["uid"], dtype=object).copy()
            traj_out = (
                np.asarray(out.non_tensor_batch["traj_uid"], dtype=object).copy()
                if "traj_uid" in out.non_tensor_batch else None
            )
            for chunk_idx, (_, dup_id) in enumerate(chunks):
                if dup_id is None:
                    continue
                start = chunk_idx * group_size
                end = start + group_size
                suffix = f"::dup_{dup_id}"
                for pos in range(start, end):
                    uid_out[pos] = f"{uid_out[pos]}{suffix}"
                    if traj_out is not None:
                        traj_out[pos] = f"{traj_out[pos]}{suffix}_{pos - start}"
                    step_info = out.non_tensor_batch.get("step_info")
                    if step_info is not None and isinstance(step_info[pos], dict):
                        step_info[pos]["turn_batch_exact_rows_duplicate"] = 1
                        step_info[pos]["turn_batch_exact_rows_duplicate_id"] = int(dup_id)
            out.non_tensor_batch["uid"] = uid_out
            if traj_out is not None:
                out.non_tensor_batch["traj_uid"] = traj_out

        out.meta_info["turn_batch_exact_train_rows_target"] = target_rows
        out.meta_info["turn_batch_exact_train_rows_actual"] = len(out)
        return out

    def _report_pregen_mismatch(self, step: int, pre_ids: torch.Tensor, live_ids: torch.Tensor) -> None:
        """Diagnostics for the pregen prompt invariant: which rows differ and where (decoded text around the first diff)."""
        try:
            if pre_ids.shape != live_ids.shape:
                print(f"[SLIC pregen] turn {step}: shape pre={tuple(pre_ids.shape)} live={tuple(live_ids.shape)}", flush=True)
                return
            diff_rows = (pre_ids != live_ids).any(dim=1).nonzero().flatten().tolist()
            print(f"[SLIC pregen] turn {step}: {len(diff_rows)}/{pre_ids.shape[0]} rows differ; rows={diff_rows[:16]}", flush=True)
            pad = self.tokenizer.pad_token_id
            for r in diff_rows[:2]:
                ta = [int(t) for t in pre_ids[r].tolist() if t != pad]
                tb = [int(t) for t in live_ids[r].tolist() if t != pad]
                a = self.tokenizer.decode(ta, skip_special_tokens=False)
                b = self.tokenizer.decode(tb, skip_special_tokens=False)
                k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                kt = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                print(f"[SLIC pregen] row {r}: tokens pre={len(ta)} live={len(tb)} first token diff at {kt}; "
                      f"first char diff at {k} (pre chars {len(a)}, live chars {len(b)})\n"
                      f"  PRE : ...{a[max(0, k-300):k+500]!r}\n  LIVE: ...{b[max(0, k-300):k+500]!r}\n"
                      f"  PRE-TAIL : {a[-400:]!r}\n  LIVE-TAIL: {b[-400:]!r}", flush=True)
        except Exception as exc:  # diagnostics must never mask the invariant
            print(f"[SLIC pregen] diagnostics failed: {exc!r}", flush=True)

    @staticmethod
    def _subset_obs(obs: Dict, idx: np.ndarray) -> Dict:
        """Row-subset of an observation dict ('text'/'image'/'anchor'...: list, ndarray, tensor or None)."""
        out = {}
        for key, value in obs.items():
            if value is None:
                out[key] = None
            elif isinstance(value, (list, tuple)):
                out[key] = [value[int(i)] for i in idx]
            elif isinstance(value, np.ndarray):
                out[key] = value[idx]
            elif torch.is_tensor(value):
                out[key] = value[torch.as_tensor(idx, device=value.device)]
            else:
                raise TypeError(f"Unsupported observation field type for {key!r}: {type(value).__name__}")
        return out

    @staticmethod
    def _expand_group_mask(group_mask: np.ndarray, batch_size: int, group_size: int) -> np.ndarray:
        expanded = np.zeros(batch_size, dtype=bool)
        for i in range(batch_size):
            expanded[i] = bool(group_mask[i // group_size])
        return expanded

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            episode_wall_clock_seconds: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            total_infos: List[List[Dict]] = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        
        for key, value in success.items():
            # Skip non-numeric metadata if any are present.
            if key == 'sampled_topologies':
                continue
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            for step_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards already include per-step penalties.
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_wall_clock_sec'] = episode_wall_clock_seconds[bs]
                    data['tool_callings'] = tool_callings[bs]
                    for key, value in success_rate.items():
                        data[key] = value
                    
                    # Add info dict for this step if available
                    if total_infos is not None and bs < len(total_infos) and step_idx < len(total_infos[bs]):
                        data['step_info'] = total_infos[bs][step_idx]

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def _collect_slic_vanilla_base_histories(
        self,
        obs,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        env_reset_kwargs,
        batch_size: int,
        group_size: int,
        num_groups: int,
        max_steps: int,
    ) -> tuple:
        """
        Pre-collect one base trajectory per group for the fixed-base baseline (continuation_policy=fixed_base).

        Only the group leaders are sampled; each leader's action is replayed in every copy of its group. On return
        the envs are at h_T; the caller restores h_0 with _restore_vanilla_base_state.

        Returns:
            base_histories: List[List[str]], base_histories[g][t] = raw action at step t
            group_done:     np.ndarray(bool), groups whose base trajectory ended before max_steps
            obs_by_depth:   observations at h_t for t = 0..T-1
            base_snaps:     base_snaps[t][i] = state of env i at h_t, t = 0..T
        """
        base_histories: List[List[str]] = [[] for _ in range(num_groups)]
        group_done = np.zeros(num_groups, dtype=bool)
        # The base trajectory needs one sample per group: the LLM sees only the group leaders (num_groups rows), and
        # each leader's action is replicated to every copy's env so all copies stay at the base state.
        leader_idx = np.arange(0, batch_size, group_size, dtype=np.int64)
        if len(leader_idx) != num_groups:
            raise RuntimeError(f"SLIC base rollout: {len(leader_idx)} leaders for {num_groups} groups.")

        obs_by_depth: List[Dict] = []   # obs at h_t for ALL envs (copies share the base state), t = 0..T-1
        # Full per-env state at every depth (0 simulations): the main loop puts every candidate back on the BASE
        # trajectory's h_{t+1}. base_snaps[t][i] = state of env i at h_t.
        base_snaps: List[list] = []
        all_idx = list(range(batch_size))
        for _step in range(max_steps):
            if group_done.all():
                break
            obs_by_depth.append(obs)
            base_snaps.append(envs.tree_capture_env_states(all_idx, obs))

            batch = self.preprocess_batch(gen_batch=gen_batch.select_idxs(leader_idx), obs=self._subset_obs(obs, leader_idx))

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            text_actions = self.tokenizer.batch_decode(
                batch_output.batch['responses'], skip_special_tokens=True
            )
            # Capture base member raw actions before envs.step may mutate the list.
            if len(text_actions) != num_groups:
                raise RuntimeError(f"SLIC base rollout: expected {num_groups} leader actions, got {len(text_actions)}.")
            raw_base_actions = list(text_actions)

            replicated_actions = [raw_base_actions[i // group_size] for i in range(batch_size)]

            obs, _, dones, _ = envs.step(replicated_actions)
            dones_np = torch_to_numpy(dones)
            if len(dones_np.shape) == 2:
                dones_np = dones_np.squeeze(1)

            for g in range(num_groups):
                if group_done[g]:
                    continue
                base_histories[g].append(raw_base_actions[g])
                if bool(dones_np[g * group_size]):
                    group_done[g] = True
        base_snaps.append(envs.tree_capture_env_states(all_idx, obs))   # h_T (or h_L for early-done groups)

        return base_histories, group_done, obs_by_depth, base_snaps

    @staticmethod
    def _restore_vanilla_base_state(envs, base_snaps, depth: int, last_obs: Dict, group_done: np.ndarray,
                                    batch_size: int, group_size: int) -> Dict:
        """Strict SLIC: put every env of every live group back on the base trajectory's state at `depth`."""
        if depth >= len(base_snaps):
            raise RuntimeError(f"SLIC base restore: no base state captured for depth {depth} "
                               f"(have {len(base_snaps)}); live groups must end with the base trajectory.")
        assignments = [(i, base_snaps[depth][i]) for i in range(batch_size) if not group_done[i // group_size]]
        if not assignments:
            return last_obs
        return envs.tree_restore_env_states(assignments, last_obs)

    def _pregenerate_vanilla_candidates(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        obs_by_depth: List[Dict],
        base_histories: List[List[str]],
        batch_size: int,
    ) -> List[DataProto]:
        """Strict SLIC: every turn's history h_t is fixed by the base trajectory, so the G
        candidates of ALL turns can be generated in ONE rollout call instead of one sequential call per turn.
        Returns one generation output per depth (rows aligned with the env batch); the main loop consumes
        pregen[_step] and checks that the prompt it would have built equals the pre-built one."""
        n_depths = max((len(h) for h in base_histories), default=0)
        if n_depths == 0:
            return []
        if len(obs_by_depth) < n_depths:
            raise RuntimeError(f"SLIC pregen: {len(obs_by_depth)} observation depths for {n_depths} base turns.")
        inputs = []
        for t in range(n_depths):
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs_by_depth[t])
            if len(batch) != batch_size:
                raise RuntimeError(f"SLIC pregen: depth {t} built {len(batch)} rows, expected {batch_size}.")
            keys = ["input_ids", "attention_mask", "position_ids"]
            nkeys = [k for k in ("raw_prompt_ids", "multi_modal_data", "raw_prompt", "tools_kwargs") if k in batch.non_tensor_batch]
            inputs.append(batch.pop(batch_keys=keys, non_tensor_batch_keys=nkeys))
        merged = DataProto.concat(inputs)
        merged.meta_info = gen_batch.meta_info
        padded, pad_size = pad_dataproto_to_divisor(merged, actor_rollout_wg.world_size)
        out = unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)
        if len(out) != n_depths * batch_size:
            raise RuntimeError(f"SLIC pregen: generated {len(out)} rows, expected {n_depths * batch_size}.")
        return out.chunk(n_depths)

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Roll out all envs for up to env.max_steps turns.

        Returns:
            (total_batch_list, episode_rewards, episode_lengths, episode_wall_clock_seconds,
             success, traj_uid, tool_callings, total_infos)
        """

        batch_size = len(gen_batch)

        # Do not pop env_kwargs: dynamic sampling may reuse gen_batch, and popping would drop task_data on retries.
        env_reset_kwargs = gen_batch.non_tensor_batch.get('env_kwargs', None)
        if "circuitop" in str(self.config.env.env_name).lower() and env_reset_kwargs is None:
            raise ValueError(
                "CircuitOp rollout requires non_tensor_batch['env_kwargs'], but it is missing."
            )
        obs, infos = envs.reset(kwargs=env_reset_kwargs)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert batch_size == lenght_obs, f"gen_batch size {batch_size} does not match obs size {lenght_obs}"

        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)

        is_slic = self._is_slic()
        group_size = int(self.config.env.rollout.n) if int(self.config.env.rollout.n) > 0 else batch_size
        # Allow tail groups (e.g., validation batch sizes not divisible by env.rollout.n).
        num_groups = (batch_size + group_size - 1) // group_size
        base_member_indices = np.array([g * group_size for g in range(num_groups)], dtype=np.int32)
        group_done = np.zeros(num_groups, dtype=bool)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        episode_wall_clock_seconds = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        slic_policy = (
            canonical_continuation_policy(self._slic_cfg_get("continuation_policy", "random"))
            if is_slic else "first"
        )

        # === Fixed-base pre-rollout ===
        # Collect one independent base trajectory per group first,
        # then sample G candidates from each fixed history context h_t.
        # The base trajectory is NOT part of the G training candidates.
        vanilla_base_histories: List[List[str]] = []
        vanilla_pregen: Optional[List[DataProto]] = None
        vanilla_base_snaps: List[list] = []
        if is_slic and is_train and slic_policy == "fixed_base":
            vanilla_base_histories, _, vanilla_obs_by_depth, vanilla_base_snaps = self._collect_slic_vanilla_base_histories(
                obs=obs,
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                env_reset_kwargs=env_reset_kwargs,
                batch_size=batch_size,
                group_size=group_size,
                num_groups=num_groups,
                max_steps=self.config.env.max_steps,
            )
            # Pre-rollout advanced obs to h_T; put every env back on h_0 (captured state, 0 sims).
            obs = self._restore_vanilla_base_state(envs, vanilla_base_snaps, 0, obs,
                                                   np.zeros(num_groups, dtype=bool), batch_size, group_size)
            vanilla_pregen = self._pregenerate_vanilla_candidates(
                gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, obs_by_depth=vanilla_obs_by_depth,
                base_histories=vanilla_base_histories, batch_size=batch_size)

        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            step_wall_clock_start = time.perf_counter()
            if is_slic and is_train:
                active_masks = self._expand_group_mask(~group_done, batch_size=batch_size, group_size=group_size)
            else:
                active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # ========================= LLM sampling =========================
            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)

            if vanilla_pregen is not None:
                # Strict SLIC: candidates for this turn were generated up front from the same fixed history.
                if _step >= len(vanilla_pregen):
                    raise RuntimeError(f"SLIC pregen: no pre-generated candidates for turn {_step}.")
                batch_output = vanilla_pregen[_step]
                pre_ids, live_ids = batch_output.batch["prompts"], batch_input.batch["input_ids"]
                # Done groups (base trajectory ended) are neither restored nor trained (padding rows): compare live rows only.
                _live = torch.as_tensor(np.asarray(active_masks, dtype=bool))
                if pre_ids.shape != live_ids.shape or not torch.equal(pre_ids[_live], live_ids[_live]):
                    self._report_pregen_mismatch(_step, pre_ids[_live], live_ids[_live])
                    raise RuntimeError(f"SLIC pregen: prompt mismatch at turn {_step} — the replayed history "
                                       "does not reproduce the base-rollout observation.")
                batch = batch.union(batch_output)
            else:
                batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
                batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
                batch = batch.union(batch_output)

            # Attach uid metadata after sampling is finalized
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            step_wall_clock_sec = time.perf_counter() - step_wall_clock_start
            episode_wall_clock_seconds[active_masks] += step_wall_clock_sec

            # Store is_action_valid for this step
            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=np.int32)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=np.int32)

            # Add rollout timing metadata for this step.
            step_wall_clock_arr = np.zeros(batch_size, dtype=np.float32)
            step_wall_clock_arr[active_masks] = step_wall_clock_sec
            batch.non_tensor_batch['rollout_step_wall_clock_sec'] = step_wall_clock_arr
            batch.non_tensor_batch['rollout_episode_wall_clock_sec_so_far'] = episode_wall_clock_seconds.copy()

            for i, info in enumerate(infos):
                if isinstance(info, dict):
                    info['rollout_step_index'] = int(_step)
                    info['rollout_step_wall_clock_sec'] = float(step_wall_clock_arr[i])
                    info['rollout_episode_wall_clock_sec_so_far'] = float(episode_wall_clock_seconds[i])

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            
            # Accumulate rewards for active environments (penalty will be applied later in trainer)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            rewards_np = torch_to_numpy(rewards)

            turn_selected_indices = None
            turn_candidate_scores = None
            turn_all_invalid_fallback = None
            turn_group_terminal = None
            if is_slic and is_train:
                if slic_policy == "fixed_base":
                    # Vanilla SLIC: no candidate selection; group terminal is
                    # determined by the pre-collected base trajectory length.
                    turn_selected_indices = np.array(
                        [g * group_size for g in range(num_groups)], dtype=np.int64
                    )
                    turn_candidate_scores = np.full(batch_size, np.nan, dtype=np.float64)
                    turn_all_invalid_fallback = np.zeros(num_groups, dtype=bool)
                    turn_group_terminal = np.array(
                        [_step >= len(vanilla_base_histories[g]) - 1 for g in range(num_groups)],
                        dtype=bool,
                    )
                    selected_mask = np.zeros(batch_size, dtype=np.int32)
                    selected_row_for_group = np.full(batch_size, -1, dtype=np.int32)
                    selected_score_for_group = np.full(batch_size, np.nan, dtype=np.float32)
                    group_all_invalid = np.zeros(batch_size, dtype=np.int32)
                    group_terminal = np.zeros(batch_size, dtype=np.int32)
                    for g in range(num_groups):
                        start = g * group_size
                        end = min(start + group_size, batch_size)
                        group_terminal[start:end] = int(turn_group_terminal[g])
                        for idx in range(start, end):
                            if isinstance(infos[idx], dict):
                                infos[idx]['slic_continuation_score'] = float('nan')
                                infos[idx]['slic_continuation_selected'] = 0
                                infos[idx]['slic_continuation_selected_row'] = -1
                                infos[idx]['slic_continuation_policy'] = "fixed_base"
                                infos[idx]['slic_continuation_score_mode'] = "none"
                                infos[idx]['slic_continuation_all_invalid_fallback'] = 0
                                infos[idx]['turn_group_terminal'] = int(turn_group_terminal[g])
                else:
                    turn_selected_indices, turn_candidate_scores, turn_all_invalid_fallback, turn_group_terminal = (
                        self._select_slic_continuation_indices(
                            infos=infos,
                            rewards=rewards_np,
                            batch_size=batch_size,
                            group_size=group_size,
                            group_done=group_done,
                        )
                    )
                    selected_mask = np.zeros(batch_size, dtype=np.int32)
                    selected_row_for_group = np.full(batch_size, -1, dtype=np.int32)
                    selected_score_for_group = np.full(batch_size, np.nan, dtype=np.float32)
                    group_all_invalid = np.zeros(batch_size, dtype=np.int32)
                    group_terminal = np.zeros(batch_size, dtype=np.int32)
                    for g, selected_idx in enumerate(turn_selected_indices):
                        start = g * group_size
                        end = min(start + group_size, batch_size)
                        if selected_idx < 0:
                            continue
                        selected_mask[int(selected_idx)] = 1
                        selected_row_for_group[start:end] = int(selected_idx)
                        selected_score_for_group[start:end] = float(turn_candidate_scores[int(selected_idx)])
                        group_all_invalid[start:end] = int(turn_all_invalid_fallback[g])
                        group_terminal[start:end] = int(turn_group_terminal[g])
                        for idx in range(start, end):
                            if isinstance(infos[idx], dict):
                                infos[idx]['slic_continuation_score'] = float(turn_candidate_scores[idx])
                                infos[idx]['slic_continuation_selected'] = int(idx == int(selected_idx))
                                infos[idx]['slic_continuation_selected_row'] = int(selected_idx)
                                infos[idx]['slic_continuation_policy'] = canonical_continuation_policy(
                                    self._slic_cfg_get("continuation_policy", "random")
                                )
                                infos[idx]['slic_continuation_score_mode'] = str(
                                    self._slic_cfg_get("continuation_score", "perf_score_post")
                                )
                                infos[idx]['slic_continuation_all_invalid_fallback'] = int(
                                    turn_all_invalid_fallback[g]
                                )
                                infos[idx]['turn_group_terminal'] = int(turn_group_terminal[g])
                batch.non_tensor_batch['slic_continuation_score'] = turn_candidate_scores.astype(np.float32)
                batch.non_tensor_batch['slic_continuation_selected'] = selected_mask
                batch.non_tensor_batch['slic_continuation_selected_row'] = selected_row_for_group
                batch.non_tensor_batch['slic_continuation_selected_score'] = selected_score_for_group
                batch.non_tensor_batch['slic_continuation_all_invalid_fallback'] = group_all_invalid
                batch.non_tensor_batch['turn_group_terminal'] = group_terminal

            batch.non_tensor_batch['rewards'] = rewards_np.astype(object)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            batch.non_tensor_batch['turn_index'] = np.full(batch_size, _step, dtype=np.int32)

            # Propagate env_kwargs/extra_info from gen_batch so they appear in
            # rollout JSONL dumps (needed for task_id, topology, circuit_name).
            if 'env_kwargs' in gen_batch.non_tensor_batch and 'env_kwargs' not in batch.non_tensor_batch:
                batch.non_tensor_batch['env_kwargs'] = gen_batch.non_tensor_batch['env_kwargs']
            if 'extra_info' in gen_batch.non_tensor_batch and 'extra_info' not in batch.non_tensor_batch:
                batch.non_tensor_batch['extra_info'] = gen_batch.non_tensor_batch['extra_info']

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            if is_slic and is_train:
                # SLIC history alignment:
                # - advance the shared base-action history for each group,
                # - update group termination,
                # - put every candidate of a live group on the selected state, so next-turn candidates share h_t.
                if slic_policy == "fixed_base":
                    # fixed_base: the history is the pre-collected base trajectory, independent of the G candidates.
                    for g in range(num_groups):
                        if group_done[g]:
                            continue
                        if _step + 1 >= len(vanilla_base_histories[g]):   # the group ends with its base trajectory
                            group_done[g] = True
                else:
                    for g, base_idx in enumerate(base_member_indices):
                        if group_done[g]:
                            continue
                        selected_idx = int(turn_selected_indices[g]) if turn_selected_indices is not None else int(base_idx)
                        if bool(turn_group_terminal[g]) or bool(dones[selected_idx]):
                            group_done[g] = True

                is_done = self._expand_group_mask(group_done, batch_size=batch_size, group_size=group_size)
                if group_done.all():
                    break
                # continuation sync: every env of a live group adopts the selected branch's state (0 extra simulations).
                # Done groups are excluded: their envs keep receiving batch-wide step() calls and drift past the episode
                # end; their rows are padding.
                if slic_policy == "fixed_base":
                    obs = self._restore_vanilla_base_state(envs, vanilla_base_snaps, _step + 1, next_obs,
                                                           group_done, batch_size, group_size)
                else:
                    obs = envs.sync_group_states(turn_selected_indices, group_size, next_obs,
                                                 skip_groups=group_done)
            else:
                # Update done states (normal independent inference, including validation)
                is_done = np.logical_or(is_done, dones)
                # Update observations for next step
                obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, episode_wall_clock_seconds, success, traj_uid, tool_callings, total_infos
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_episode_wall_clock_seconds (np.ndarray): Wall-clock time per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
            total_tool_callings (np.ndarray): Tool calls per episode.
            total_infos_all (List[Dict]): Per-step environment infos.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_episode_wall_clock_seconds = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_infos_all = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            try_count += 1

            batch_list, episode_rewards, episode_lengths, episode_wall_clock_seconds, success, traj_uid, tool_callings, infos = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, episode_wall_clock_seconds, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                episode_wall_clock_seconds=episode_wall_clock_seconds,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_episode_wall_clock_seconds.append(episode_wall_clock_seconds)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            total_infos_all.extend(infos)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_episode_wall_clock_seconds = np.concatenate(total_episode_wall_clock_seconds, axis=0)
        
        # Handle success dict: merge sets, concatenate arrays
        merged_success = {}
        for key in total_success[0].keys():
            if key == 'sampled_topologies':
                # Merge sets from all batches
                merged_success[key] = set().union(*[success[key] for success in total_success])
            else:
                # Concatenate numeric arrays
                merged_success[key] = np.concatenate([success[key] for success in total_success], axis=0)
        total_success = merged_success
        
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_episode_wall_clock_seconds, total_success, total_traj_uid, total_tool_callings, total_infos_all

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        if is_train and self._is_mt_grpo_tree():
            from agent_system.multi_turn_rollout.mt_grpo_tree_rollout import mt_grpo_tree_multi_turn_loop
            total_batch_list, total_episode_rewards, total_episode_lengths, \
                total_episode_wall_clock_seconds, total_success, total_traj_uid, \
                totoal_tool_callings, total_infos = mt_grpo_tree_multi_turn_loop(
                    self, gen_batch, actor_rollout_wg, envs)
            out = self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                episode_wall_clock_seconds=total_episode_wall_clock_seconds,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
                total_infos=total_infos,
            )
            counts = getattr(self, "_mt_grpo_tree_last_counts", {})
            for key in ("generated", "idle_slot_steps", "rows", "trajectories", "tasks"):
                if key in counts:
                    out.meta_info[f"mt_grpo_tree/{key}"] = int(counts[key])
            return out
        if self._is_slic() and self.config.algorithm.filter_groups.enable and is_train:
            raise RuntimeError(
                "SLIC requires fixed per-turn groups; disable algorithm.filter_groups.enable."
            )
            
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_episode_wall_clock_seconds, total_success, total_traj_uid, totoal_tool_callings, total_infos = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_episode_wall_clock_seconds, total_success, total_traj_uid, totoal_tool_callings, total_infos = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_episode_wall_clock_seconds)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            episode_wall_clock_seconds=total_episode_wall_clock_seconds,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            total_infos=total_infos,
        )
        if is_train and self._is_slic() and not self._as_bool(self._slic_cfg_get("skip_degenerate_groups", False)):
            gen_batch_output = self._apply_turn_batch_exact_rows(gen_batch_output)
        
        return gen_batch_output
