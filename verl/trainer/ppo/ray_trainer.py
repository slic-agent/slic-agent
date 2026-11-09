# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.fs import resolve_ppo_checkpoint_model_path
from verl.utils.fsdp_checkpoint_loader import detect_checkpoint_format
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch
from pathlib import Path as _P
_REPO_ROOT = str(_P(__file__).resolve().parents[2])


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    SLIC = "slic"
    MT_GRPO_TREE = "mt_grpo_tree"


def _format_progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[unknown]"
    completed = max(0, min(int(completed), int(total)))
    filled = int(round((completed / total) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _extract_task_id(env_kw) -> Optional[str]:
    if not isinstance(env_kw, dict):
        return None
    task_id = env_kw.get("task_id")
    if task_id:
        return str(task_id)
    task_data = env_kw.get("task_data")
    if isinstance(task_data, dict):
        for key in ("task_id", "id", "base_task_id"):
            value = task_data.get(key)
            if value:
                return str(value)
    return None


def _format_task_preview(env_kwargs, max_items: int = 4) -> str:
    labels = []
    for env_kw in list(env_kwargs if env_kwargs is not None else []):
        task_id = _extract_task_id(env_kw)
        if task_id:
            labels.append(task_id)
            continue
        if isinstance(env_kw, dict):
            topology = env_kw.get("topology")
            fallback = str(topology) if topology else ""
            if fallback:
                labels.append(fallback)

    if not labels:
        return "unavailable"

    preview = labels[:max_items]
    if len(labels) > max_items:
        preview.append(f"... (+{len(labels) - max_items} more)")
    return ", ".join(preview)


def _format_metric_value(metrics: Dict[str, object], key: str, digits: int = 3) -> Optional[str]:
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return f"{float(value):.{digits}f}"
    return None


def _format_training_metric_summary(metrics: Dict[str, object]) -> str:
    metric_specs = [
        ("episode_reward", "episode/reward/mean", 3),
        ("episode_len", "episode/length/mean", 2),
        ("score", "critic/score/mean", 3),
        ("step_s", "timing_s/step", 2),
        ("tok/s/gpu", "perf/throughput", 1),
    ]
    parts = []
    for label, key, digits in metric_specs:
        formatted = _format_metric_value(metrics, key, digits)
        if formatted is not None:
            parts.append(f"{label}={formatted}")
    return " ".join(parts) if parts else "metrics=unavailable"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def _is_simulation_failure_error_type(error_type: str) -> bool:
    """True if error_type indicates simulation/design failure (not format/validation)."""
    if not error_type:
        return False
    et = str(error_type).lower()
    return (
        "simulation" in et
        or "design_error" in et
        or "convergence" in et
    )


def compute_valid_action_metrics(data: DataProto) -> dict:
    """Compute episode validity metrics without applying any penalty."""
    total_invalid = 0
    total_actions = 0
    total_format_invalid = 0
    total_sim_invalid = 0

    step_infos = data.non_tensor_batch.get('step_info')

    for i in range(len(data)):
        data_item = data[i]
        action_valids = data_item.non_tensor_batch.get('is_action_valid')
        if action_valids is None:
            continue

        if isinstance(action_valids, np.ndarray) and action_valids.size > 1:
            is_current_step_invalid = (action_valids[-1] == 0)
            num_invalid_in_episode = int(np.sum(1 - action_valids))
            num_total = int(len(action_valids))
        else:
            is_current_step_invalid = (float(action_valids) == 0)
            num_invalid_in_episode = int(is_current_step_invalid)
            num_total = 1

        total_invalid += num_invalid_in_episode
        total_actions += num_total

        if is_current_step_invalid:
            error_type = ""
            if step_infos is not None and i < len(step_infos):
                si = step_infos[i]
                if isinstance(si, dict):
                    error_type = si.get("error_type", "") or ""
            if _is_simulation_failure_error_type(error_type):
                total_sim_invalid += 1
            else:
                total_format_invalid += 1

    valid_action_ratio = 1.0 - (total_invalid / max(total_actions, 1))
    return {
        "episode/valid_action_ratio": valid_action_ratio,
        "episode/invalid_format_count": total_format_invalid,
        "episode/invalid_simulation_count": total_sim_invalid,
    }


def apply_invalid_action_penalty(
    data: DataProto,
    invalid_action_penalty_coef: float = 0.1,
    simulation_failure_penalty_coef: Optional[float] = None,
    invalid_action_penalty_mode: str = "last_token",
):
    """
    Apply penalty for invalid action at EACH step individually.

    Two penalty levels:
    - Format/validation (parsing, non-English/weird symbols, semantic validation):
      uses invalid_action_penalty_coef (larger).
    - Simulation failure (valid action led to invalid design / sim failed):
      uses simulation_failure_penalty_coef (smaller). If None, same as invalid_action_penalty_coef.
    """
    if simulation_failure_penalty_coef is None:
        simulation_failure_penalty_coef = invalid_action_penalty_coef

    reward_tensor = data.batch['token_level_scores']
    step_rewards = data.batch['step_rewards'] if 'step_rewards' in data.batch.keys() else None
    step_infos = data.non_tensor_batch.get('step_info')
    responses = data.batch['responses']
    response_length = responses.size(1)
    loss_mask = data.batch['loss_mask'] if 'loss_mask' in data.batch.keys() else None
    attention_mask = data.batch['attention_mask'] if 'attention_mask' in data.batch.keys() else None

    if invalid_action_penalty_mode not in {"last_token", "per_token"}:
        raise ValueError(
            f"Unsupported invalid_action_penalty_mode={invalid_action_penalty_mode!r}. "
            "Expected one of: {'last_token', 'per_token'}."
        )

    total_invalid_actions = 0
    total_actions = 0
    total_format_invalid = 0
    total_sim_invalid = 0

    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        # Get validity flags for ALL steps in this episode
        action_valids = data_item.non_tensor_batch.get('is_action_valid')
        if action_valids is None:
            # Environment does not provide action validity — skip penalty
            continue

        # Determine if THIS step's action is invalid (check last entry in history)
        if isinstance(action_valids, np.ndarray) and action_valids.size > 1:
            is_current_step_invalid = (action_valids[-1] == 0)  # Last action = current step
            num_invalid_in_episode = int(np.sum(1 - action_valids))  # For metrics
            num_total = int(len(action_valids))
        else:
            # Single value (single-step episode or scalar)
            is_current_step_invalid = (float(action_valids) == 0)
            num_invalid_in_episode = int(is_current_step_invalid)
            num_total = 1

        total_invalid_actions += num_invalid_in_episode
        total_actions += num_total

        # Apply penalty ONLY for THIS step if THIS step's action is invalid
        if is_current_step_invalid:
            # Resolve error_type for this row (batch may be per-step rows)
            error_type = ""
            if step_infos is not None and i < len(step_infos):
                si = step_infos[i]
                if isinstance(si, dict):
                    error_type = si.get("error_type", "") or ""

            if _is_simulation_failure_error_type(error_type):
                penalty = simulation_failure_penalty_coef
                total_sim_invalid += 1
            else:
                penalty = invalid_action_penalty_coef
                total_format_invalid += 1

            if invalid_action_penalty_mode == "last_token":
                reward_tensor[i, valid_response_length - 1] -= penalty
                effective_penalty = penalty
            else:
                if loss_mask is not None:
                    token_mask = loss_mask[i, -response_length:]
                elif attention_mask is not None:
                    token_mask = attention_mask[i, -response_length:]
                else:
                    raise RuntimeError(
                        "apply_invalid_action_penalty requires either batch['loss_mask'] "
                        "or batch['attention_mask'] to construct the response token mask."
                    )
                token_mask = token_mask.to(dtype=reward_tensor.dtype)
                reward_tensor[i] -= token_mask * penalty
                effective_penalty = float(token_mask.sum().item()) * penalty
            if step_rewards is not None:
                step_rewards[i] -= effective_penalty

    valid_action_ratio = 1.0 - (total_invalid_actions / max(total_actions, 1))
    metrics = {
        "episode/valid_action_ratio": valid_action_ratio,
        "episode/invalid_format_count": total_format_invalid,
        "episode/invalid_simulation_count": total_sim_invalid,
    }
    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    step_wise_reward_key="perf_score_post",
    step_wise_use_fixed_invalid_turn_reward=False,
    step_wise_invalid_turn_reward=-0.1,
    step_wise_advantage_mode="joint",
    step_wise_validity_weight=1.0,
    step_wise_performance_weight=1.0,
    step_wise_sepvp_weight_normalize=False,
    mt_grpo_alpha=1.0,
    mt_grpo_reward_key="perf_score_post",
    mt_grpo_use_fixed_invalid_turn_reward=True,
    mt_grpo_invalid_turn_reward=-0.1,
    mt_grpo_outcome_source="env_reward",
    **kwargs,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.MT_GRPO_TREE:
        # MT-GRPO full tree: rows come from mt_grpo_tree_rollout with node / parent / terminal / outcome fields;
        # their absence means the batch did not come from the tree loop -> error, never a silent fallback.
        for key in ("mtg_node_id", "mtg_parent_id", "mtg_is_terminal", "mtg_outcome_reward", "step_info", "uid"):
            if key not in data.non_tensor_batch:
                raise RuntimeError(f"adv_estimator=mt_grpo_tree but non_tensor_batch has no '{key}'; "
                                   "training rollouts must go through mt_grpo_tree_multi_turn_loop.")
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        nt = data.non_tensor_batch
        pad = np.asarray(nt.get("is_padding_sample", np.zeros(len(data), dtype=bool)), dtype=bool)
        real = np.flatnonzero(~pad)
        advantages = torch.zeros_like(grpo_calculation_mask, dtype=torch.float32)
        if len(real):
            adv_real, _ = core_algos.compute_mt_grpo_tree_advantage(
                response_mask=grpo_calculation_mask[real],
                index=np.asarray(nt["uid"], dtype=object)[real],
                node_id=np.asarray(nt["mtg_node_id"], dtype=object)[real],
                parent_id=np.asarray(nt["mtg_parent_id"], dtype=object)[real],
                is_terminal=np.asarray(nt["mtg_is_terminal"], dtype=bool)[real],
                outcome_reward=np.asarray(nt["mtg_outcome_reward"], dtype=np.float64)[real],
                step_info=np.asarray(nt["step_info"], dtype=object)[real],
                reward_key=mt_grpo_reward_key,
                use_fixed_invalid_turn_reward=mt_grpo_use_fixed_invalid_turn_reward,
                invalid_turn_reward=mt_grpo_invalid_turn_reward,
                alpha=mt_grpo_alpha,
                outcome_source=mt_grpo_outcome_source,
            )
            advantages[torch.as_tensor(real)] = adv_real.to(advantages.dtype)
        data.batch["advantages"] = advantages
        data.batch["returns"] = advantages.clone()
    elif adv_estimator == AdvantageEstimator.SLIC:
        # SLIC: normalize history-conditioned turn rewards extracted from step_info.
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        if "turn_index" not in data.non_tensor_batch:
            raise RuntimeError("SLIC requires non_tensor_batch['turn_index'].")
        if "traj_uid" not in data.non_tensor_batch:
            raise RuntimeError("SLIC requires non_tensor_batch['traj_uid'].")
        if "step_info" not in data.non_tensor_batch:
            raise RuntimeError("SLIC requires non_tensor_batch['step_info'].")
        advantage_kwargs = dict(
            response_mask=grpo_calculation_mask,
            index=np.asarray(data.non_tensor_batch["uid"], dtype=object),
            traj_index=np.asarray(data.non_tensor_batch["traj_uid"], dtype=object),
            turn_index=np.asarray(data.non_tensor_batch["turn_index"], dtype=np.int64),
            step_info=np.asarray(data.non_tensor_batch["step_info"], dtype=object),
            reward_key=step_wise_reward_key,
            use_fixed_invalid_turn_reward=step_wise_use_fixed_invalid_turn_reward,
            invalid_turn_reward=step_wise_invalid_turn_reward,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        if step_wise_advantage_mode == "joint":
            advantages, returns = core_algos.compute_turn_group_joint_advantage(**advantage_kwargs)
        elif step_wise_advantage_mode == "separate_validity_perf":
            _pad = data.non_tensor_batch.get("is_padding_sample")
            advantages, returns, components = core_algos.compute_turn_separate_validity_perf_advantage(
                **advantage_kwargs,
                validity_weight=step_wise_validity_weight,
                performance_weight=step_wise_performance_weight,
                is_padding_sample=None if _pad is None else np.asarray(_pad),
                weight_normalize=bool(step_wise_sepvp_weight_normalize),
            )
            for key, values in components.items():
                data.batch[key] = values
        else:
            raise ValueError(f"Unknown turn-level advantage mode: {step_wise_advantage_mode}")
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self._turn_batch_pending_roots = deque()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.SLIC,
            AdvantageEstimator.MT_GRPO_TREE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError(f"Unsupported adv_estimator: {self.config.algorithm.adv_estimator}")

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self._best_val_score = float("-inf")
        self._best_val_ckpt_step = None
        # (score, step) of the best checkpoints kept under <dir>/best/, outside the rotating pool;
        # persisted to best_meta.json and restored on resume.
        self._best_ckpt_history: list = []
        self._restore_best_ckpt_state()

    def _best_meta_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, "best_meta.json")

    def _restore_best_ckpt_state(self) -> None:
        """Restore best-checkpoint tracking across resumes, so a post-resume score becomes the new best only if it
        beats the best score saved before the resume."""
        import json as _json
        path = self._best_meta_path()
        if not os.path.isfile(path):
            return
        try:
            meta = _json.load(open(path))
            self._best_val_score = float(meta.get("best_val_score", float("-inf")))
            self._best_val_ckpt_step = meta.get("best_val_ckpt_step")
            self._best_ckpt_history = [tuple(x) for x in meta.get("history", [])]
            if meta.get("score_key") != self.config.trainer.get("val_score_key", "val/success_rate"):
                raise ValueError("Best checkpoint metric changed across resume")
            if not self._best_ckpt_history or not np.isfinite(self._best_val_score):
                raise ValueError("Best checkpoint metadata has no finite saved score")
            for score, step in self._best_ckpt_history:
                saved = os.path.join(self.config.trainer.default_local_dir, "best", f"global_step_{step}")
                if (not np.isfinite(score) or not os.path.isfile(os.path.join(saved, ".complete"))
                        or not os.path.isdir(os.path.join(saved, "actor"))):
                    raise ValueError(f"Protected best checkpoint is missing or incomplete: {saved}")
            print(f"[BestCkpt] Restored tracker: best={self._best_val_score:.4f} "
                  f"at step {self._best_val_ckpt_step}, {len(self._best_ckpt_history)} kept")
        except Exception as exc:
            raise RuntimeError(f"Cannot restore protected best checkpoint metadata: {path}") from exc

    def _persist_best_ckpt_state(self) -> None:
        import json as _json
        path = self._best_meta_path()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            _json.dump({"best_val_score": self._best_val_score,
                        "best_val_ckpt_step": self._best_val_ckpt_step,
                        "history": self._best_ckpt_history,
                        "score_key": self.config.trainer.get("val_score_key", "val/success_rate")}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _turn_batch_refill_enabled(self) -> bool:
        if self.config.algorithm.adv_estimator != AdvantageEstimator.SLIC:
            return False
        slic_cfg = self.config.algorithm.get("slic", {})
        target_rows = int(slic_cfg.get("target_train_rows", 0) or 0)
        return bool(slic_cfg.get("exact_rows_refill", False)) and target_rows > 0

    def _next_turn_batch_root(self, train_iter) -> DataProto:
        root_batch_size = int(self.config.data.train_batch_size)
        if root_batch_size <= 0:
            raise RuntimeError(f"data.train_batch_size must be positive, got {root_batch_size}.")

        parts = []
        need = root_batch_size
        while need > 0:
            if self._turn_batch_pending_roots:
                chunk = self._turn_batch_pending_roots.popleft()
            else:
                try:
                    chunk = DataProto.from_single_dict(next(train_iter))
                except StopIteration:
                    for part in reversed(parts):
                        self._turn_batch_pending_roots.appendleft(part)
                    raise

            if len(chunk) <= need:
                parts.append(chunk)
                need -= len(chunk)
            else:
                parts.append(chunk.slice(0, need))
                self._turn_batch_pending_roots.appendleft(chunk.slice(need, len(chunk)))
                need = 0

        return parts[0] if len(parts) == 1 else DataProto.concat(parts)

    def _build_turn_batch_from_root(self, root_batch: DataProto) -> DataProto:
        root_batch = root_batch.select_idxs(np.arange(len(root_batch), dtype=np.int64))
        saved_extra_info = root_batch.non_tensor_batch.get("extra_info", None)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in root_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in root_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in root_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in root_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")

        gen_batch = root_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        if saved_extra_info is not None and "extra_info" not in gen_batch.non_tensor_batch:
            gen_batch.non_tensor_batch["extra_info"] = saved_extra_info
        return gen_batch

    def _enqueue_turn_batch_truncated_roots(self, root_items, root_ids) -> None:
        if not root_ids:
            return

        chunks = [root_items[int(root_id)] for root_id in root_ids]
        requeue_batch = chunks[0] if len(chunks) == 1 else DataProto.concat(chunks)
        priority = str(
            self.config.algorithm.get("slic", {}).get("truncated_task_priority", "normal")
        ).strip().lower()
        if priority == "front":
            self._turn_batch_pending_roots.appendleft(requeue_batch)
        elif priority in {"normal", "weighted"}:
            self._turn_batch_pending_roots.append(requeue_batch)
        else:
            raise RuntimeError(
                "algorithm.step_wise.truncated_task_priority must be one of "
                f"'normal', 'front', or 'weighted', got {priority!r}."
            )

    def _filter_slic_turn_groups(self, output: DataProto, rollout_id: int):
        config = self.config.algorithm.slic
        metadata = output.non_tensor_batch
        keep, metrics = core_algos.compute_slic_group_keep_mask(
            index=metadata["uid"], turn_index=metadata["turn_index"], traj_index=metadata["traj_uid"],
            step_info=metadata["step_info"], group_size=int(self.config.env.rollout.n),
            reward_key=config.get("reward_key", "perf_score_post"),
            use_fixed_invalid_turn_reward=bool(config.get("use_fixed_invalid_turn_reward", False)),
            invalid_turn_reward=float(config.get("invalid_turn_reward", -0.1)),
            reward_tolerance=float(config.get("degenerate_reward_tolerance", 0.0)),
            is_padding_sample=metadata.get("is_padding_sample"),
        )
        if "formatted_prompt_text" in metadata:
            inputs = self._to_list(metadata["formatted_prompt_text"])
        else:
            inputs = self.tokenizer.batch_decode(output.batch["prompts"], skip_special_tokens=True)
        # skip_special_tokens drops the response padding, as in the main dump
        outputs = self.tokenizer.batch_decode(output.batch["responses"], skip_special_tokens=True)
        padding = np.asarray(metadata.get("is_padding_sample", np.zeros(len(output))), dtype=bool)
        active = np.flatnonzero(~padding)
        active_scores = core_algos._extract_step_reward_scores(
            np.asarray(metadata["step_info"], dtype=object)[active], config.get("reward_key", "perf_score_post"),
            dtype=torch.float32, device="cpu",
            use_fixed_invalid_turn_reward=bool(config.get("use_fixed_invalid_turn_reward", False)),
            invalid_turn_reward=float(config.get("invalid_turn_reward", -0.1)),
        ).tolist()
        scores = [None] * len(output)
        for row_index, score in zip(active, active_scores):
            scores[row_index] = score
        dump_dir = os.path.join(self.config.trainer.rollout_data_dir, "pre_filter",
                                f"step_{self.global_steps}", f"rollout_{rollout_id}_{time.time_ns()}")
        extra = {key: metadata[key] for key in ("uid", "traj_uid", "turn_index", "is_padding_sample",
                                               "turn_batch_refill_root_id", "turn_batch_refill_rollout_id") if key in metadata}
        extra["step_wise_signal_kept"] = keep.tolist()
        # task identity and error text exactly as the main dump writes them (verl/utils/rollout_dump_fields.py)
        if "env_kwargs" in metadata:
            extra["topology"] = [(kw or {}).get("topology", "unknown") if isinstance(kw, dict) else "unknown" for kw in metadata["env_kwargs"]]
        if "extra_info" in metadata:
            extra["task_id"] = [ei.get("task_id", "") if isinstance(ei, dict) else "" for ei in metadata["extra_info"]]
        extra["error"] = [si.get("error", "") if isinstance(si, dict) else "" for si in metadata["step_info"]]
        self._dump_generations(inputs, outputs, scores, {"step_info": metadata["step_info"]}, dump_dir,
                               extra_metadata=extra)
        return output.select_idxs(np.flatnonzero(keep)), metrics

    def _collect_turn_batch_refill_rollouts(
        self,
        first_root_batch: DataProto,
        first_gen_batch: DataProto,
        train_iter,
        actor_rollout_wg,
        envs,
    ):
        slic_cfg = self.config.algorithm.get("slic", {})
        target_rows = int(slic_cfg.get("target_train_rows", 0) or 0)
        if target_rows <= 0:
            raise RuntimeError("SLIC refill requires algorithm.step_wise.target_train_rows > 0.")
        skip_degenerate = bool(slic_cfg.get("skip_degenerate_groups", False))
        max_rollouts = int(slic_cfg.get("max_refill_rollouts", 8))

        output_parts = []
        eligible_parts = []
        filter_metrics = defaultdict(int)
        root_items = []
        sampled_env_kwargs = []

        def add_rollout(root_batch: DataProto, gen_batch: DataProto) -> None:
            rollout_id = len(output_parts)
            root_offset = len(root_items)
            for root_idx in range(len(root_batch)):
                root_items.append(root_batch.slice(root_idx, root_idx + 1))

            output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=True,
            )
            if len(output) == 0:
                raise RuntimeError("SLIC refill rollout produced an empty DataProto.")
            if "uid" not in output.non_tensor_batch:
                raise RuntimeError("SLIC refill requires rollout output non_tensor_batch['uid'].")

            uid_to_local_root = {}
            root_ids = []
            for uid in np.asarray(output.non_tensor_batch["uid"], dtype=object):
                if uid not in uid_to_local_root:
                    uid_to_local_root[uid] = len(uid_to_local_root)
                local_root = uid_to_local_root[uid]
                if local_root >= len(root_batch):
                    raise RuntimeError(
                        "SLIC refill saw more rollout uid groups than root tasks "
                        f"({local_root + 1} > {len(root_batch)})."
                    )
                root_ids.append(root_offset + local_root)

            output.non_tensor_batch["turn_batch_refill_rollout_id"] = np.full(
                len(output), rollout_id, dtype=np.int32
            )
            output.non_tensor_batch["turn_batch_refill_root_id"] = np.asarray(root_ids, dtype=np.int64)
            output_parts.append(output)
            if skip_degenerate:
                eligible, statistics = self._filter_slic_turn_groups(output, rollout_id)
                eligible_parts.append(eligible)
                for key, value in statistics.items():
                    filter_metrics[key] += value
            else:
                eligible_parts.append(output)

            sampled_env_kwargs.extend(list(root_batch.non_tensor_batch.get("env_kwargs", [])))

        add_rollout(first_root_batch, first_gen_batch)
        eligible = eligible_parts[0]
        selected = self.traj_collector._apply_turn_batch_exact_rows(eligible) if len(eligible) else eligible

        # `refill_shortfall=proceed` trains on whatever the bounded refill produced (used with target_train_rows >= the
        # rollout's maximum so that no turn group is truncated); `raise` stops with an error instead.
        shortfall = str(slic_cfg.get("refill_shortfall", "raise")).strip().lower()
        while len(selected) < target_rows:
            # The rollout cap applies with or without skip_degenerate_groups: filtering changes which rows are kept,
            # not how many rollouts are generated.
            if len(output_parts) >= max_rollouts:
                if shortfall == "proceed":
                    break
                raise RuntimeError(
                    f"Turn-level group refill reached max_refill_rollouts={max_rollouts}: "
                    f"only {len(selected)}/{target_rows} informative rows; raw rollouts saved under pre_filter."
                )
            try:
                root_batch = self._next_turn_batch_root(train_iter)
            except StopIteration:
                break
            gen_batch = self._build_turn_batch_from_root(root_batch)
            add_rollout(root_batch, gen_batch)
            eligible = DataProto.concat(eligible_parts)
            selected = self.traj_collector._apply_turn_batch_exact_rows(eligible) if len(eligible) else eligible

        generated_root_ids = np.asarray(eligible.non_tensor_batch["turn_batch_refill_root_id"], dtype=np.int64)
        selected_root_ids = np.asarray(selected.non_tensor_batch["turn_batch_refill_root_id"], dtype=np.int64)
        generated_counts = {
            int(root_id): int(np.sum(generated_root_ids == root_id))
            for root_id in np.unique(generated_root_ids)
        }
        selected_counts = {
            int(root_id): int(np.sum(selected_root_ids == root_id))
            for root_id in np.unique(selected_root_ids)
        }
        requeue_root_ids = [
            root_id
            for root_id, generated_count in generated_counts.items()
            if selected_counts.get(root_id, 0) < generated_count
        ]
        self._enqueue_turn_batch_truncated_roots(root_items, requeue_root_ids)

        metrics = {
            "slic/train_rows": len(selected),
            "slic/generated_rows": sum(len(part) for part in output_parts),
        }
        if skip_degenerate:
            metrics["slic/filter_skipped_group_fraction"] = filter_metrics["skipped_groups"] / max(filter_metrics["groups"], 1)
        current_task_preview = _format_task_preview(np.asarray(sampled_env_kwargs, dtype=object))
        return selected, metrics, current_task_preview

    def _validate_config(self):
        from verl.utils.circuit_checkpoint import validate_circuit_checkpoint_policy

        config = self.config
        validate_circuit_checkpoint_policy(config)
        turn_config = config.algorithm.get("slic", {})
        turn_mode = turn_config.get("advantage_mode", "joint")
        if turn_mode not in ("joint", "separate_validity_perf"):
            raise ValueError(f"Unknown turn-level advantage mode: {turn_mode}")
        if turn_mode == "separate_validity_perf":
            if not turn_config.get("use_fixed_invalid_turn_reward", False):
                raise ValueError("Separate turn-level normalization requires fixed failure handling.")
            if config.algorithm.adv_estimator != AdvantageEstimator.SLIC:
                raise ValueError("Separate validity/performance normalization requires SLIC.")
            weights = [float(turn_config.get(key, 1.0)) for key in ("validity_weight", "performance_weight")]
            if any(not np.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
                raise ValueError("Separate turn-level weights must be finite, nonnegative, and not both zero.")
            if turn_config.get("skip_degenerate_groups", False):
                # With fixed failure rewards (<0) and perf_score_post (>=0), "all group rewards equal" holds iff both
                # the validity and performance channels are flat, so the scalar keep-mask is the two-channel zero-signal
                # test. The tolerance must stay below the failure gap, or a mixed group would be dropped as flat.
                gap = abs(float(turn_config.get("invalid_turn_reward", -0.1)))
                if float(turn_config.get("degenerate_reward_tolerance", 0.0)) >= gap:
                    raise ValueError("SepVP turn-group filtering needs degenerate_reward_tolerance below the failure gap.")
                if turn_config.get("reward_key", "perf_score_post") != "perf_score_post":
                    raise ValueError("SepVP turn-group filtering requires reward_key=perf_score_post.")
        if turn_config.get("skip_degenerate_groups", False):
            if not self._turn_batch_refill_enabled():
                raise ValueError("Turn-level group filtering requires SLIC with exact_rows_refill and target_train_rows > 0.")
            if turn_config.get("exact_rows_fill_mode", "none") != "none":
                raise ValueError("Turn-level group filtering cannot use duplicate row filling.")
            if not config.trainer.get("rollout_data_dir"):
                raise ValueError("Turn-level group filtering requires rollout_data_dir for pre-filter raw dumps.")
            group_size = int(config.env.rollout.n)
            tolerance = float(turn_config.get("degenerate_reward_tolerance", 0.0))
            if group_size < 2 or int(turn_config.target_train_rows) % group_size:
                raise ValueError("Turn-level group filtering requires whole candidate groups of size >= 2.")
            if int(turn_config.get("max_refill_rollouts", 8)) < 1 or not np.isfinite(tolerance) or tolerance < 0:
                raise ValueError("Turn-level group refill limit must be positive and reward tolerance finite/nonnegative.")
            if str(turn_config.get("refill_shortfall", "raise")).strip().lower() not in ("raise", "proceed"):
                raise ValueError("algorithm.step_wise.refill_shortfall must be 'raise' or 'proceed'.")
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
            "minibatch-token-mean",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO,
                AdvantageEstimator.SLIC,
            ], "only GRPO-family estimators are tested for multi-turn with tool"


    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        val_sampler = None
        val_k = self.config.data.get("val_stratified_val_k", None)
        test_k = self.config.data.get("val_stratified_test_k", None)
        if val_k is not None or test_k is not None:
            from verl.utils.dataset.stratified_val_sampler import (
                SplitStratifiedCycleSampler,
                build_split_and_topology_labels,
            )
            split_labels, topo_labels = build_split_and_topology_labels(self.val_dataset)
            samples_per_split = {}
            if val_k is not None:
                samples_per_split["validation"] = int(val_k)
            if test_k is not None:
                samples_per_split["test"] = int(test_k)
            val_sampler = SplitStratifiedCycleSampler(
                split_labels=split_labels,
                topology_labels=topo_labels,
                samples_per_split=samples_per_split,
                default_k=3,
                seed=int(OmegaConf.select(self.config, "env.seed", default=42)),
                fixed=bool(self.config.data.get("val_stratified_fixed", False)),
            )
            val_batch_size = len(val_sampler)
            summary = val_sampler.summary()
            print(f"[Val Sampler] Split-stratified cycle sampler: {summary}")

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            sampler=val_sampler,
        )
        self.val_batch_size = val_batch_size

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path, extra_metadata=None):
        """Dump rollout/validation samples as JSONL (fields: verl/utils/rollout_dump_fields.py)."""
        import numpy as np
        import json
        from verl.utils.rollout_dump_fields import tidy_dump_row
        
        def convert_to_json_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        if len(outputs) != n or len(scores) != n:
            raise ValueError(f"Rollout dump row mismatch: inputs={n}, outputs={len(outputs)}, scores={len(scores)}")
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        # Add reward extra info
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        
        # Flatten step_info to extract environment fields as top-level keys
        if 'step_info' in reward_extra_infos_dict:
            step_info_list = reward_extra_infos_dict['step_info']
            
            if len(step_info_list) == n:
                # Fields to extract from step_info
                fields_to_extract = [
                    'is_stop_action',
                    'performance_success',
                    # Common environment fields
                    'gain_dB',
                    'power_uW',
                    'bandwidth_MHz',
                    'bandwidth_kind',
                    'inband_flatness_db',
                    'phase_margin_status',
                    'phase_margin_deg',
                    'differential_gain_db',
                    'offset_v',
                    'cmrr_db',
                    'psrr_plus_db',
                    'psrr_minus_db',
                    'initial_gain_dB',
                    'initial_power_uW',
                    'is_action_valid',
                    'error',
                    'error_type',
                    'perf_score_post',
                    'perf_score_pre',
                    'goal',
                    'done',
                    'rollout_step_index',
                    # Resulting state for dataset augmentation
                    'resulting_params',
                    'resulting_metrics',
                    # Goal thresholds (from step_info['goal']) for debugging success vs metrics
                    'goal_gain_min',
                    'goal_power_max',
                    'goal_bandwidth_min',
                    'goal_phase_margin_min_deg',
                    'goal_phase_margin_max_deg',
                    'goal_offset_abs_max_v',
                    'goal_cmrr_min_db',
                    'goal_psrr_plus_min_db',
                    'goal_psrr_minus_min_db',
                ]
                # Keys inside step_info['goal'] for goal_* fields
                goal_key_map = {
                    'goal_gain_min': 'gain_min',
                    'goal_power_max': 'power_max',
                    'goal_bandwidth_min': 'bandwidth_min',
                    'goal_phase_margin_min_deg': 'phase_margin_min_deg',
                    'goal_phase_margin_max_deg': 'phase_margin_max_deg',
                    'goal_offset_abs_max_v': 'offset_abs_max_v',
                    'goal_cmrr_min_db': 'cmrr_min_db',
                    'goal_psrr_plus_min_db': 'psrr_plus_min_db',
                    'goal_psrr_minus_min_db': 'psrr_minus_min_db',
                }

                # Extract each field from step_info dicts
                for field in fields_to_extract:
                    if field not in base_data:  # Don't override if already exists at top level
                        base_data[field] = []
                        for i in range(n):
                            step_info_dict = step_info_list[i]
                            if isinstance(step_info_dict, dict):
                                if field in goal_key_map:
                                    g = step_info_dict.get('goal') or {}
                                    base_data[field].append(g.get(goal_key_map[field]))
                                elif field in step_info_dict:
                                    base_data[field].append(step_info_dict[field])
                                else:
                                    base_data[field].append(None)
                            else:
                                base_data[field].append(None)
                if "constraint_specs" not in base_data:
                    base_data["constraint_specs"] = []
                    for i in range(n):
                        step_info_dict = step_info_list[i]
                        if isinstance(step_info_dict, dict):
                            g = step_info_dict.get("goal") or {}
                            base_data["constraint_specs"].append(g.get("constraint_specs"))
                        else:
                            base_data["constraint_specs"].append(None)
        
        # Add extra metadata (topology, circuit_name, etc.)
        if extra_metadata:
            for k, v in extra_metadata.items():
                if len(v) == n:
                    base_data[k] = v

        # Derive per-spec pass flags when CircuitOp metric/goal fields are available.
        try:
            from agent_system.environments.env_package.circuitop.circuitop.rewards import (
                apply_positive_gain_gate_to_constraint_results,
                evaluate_constraint_specs,
            )

            spec_order = [
                "gain",
                "power",
                "bandwidth",
                "phase_margin",
                "offset",
                "cmrr",
                "psrr_plus",
                "psrr_minus",
            ]
            pass_keys = {spec: f"{spec}_pass" for spec in spec_order}

            if all(k not in base_data for k in pass_keys.values()):
                derived_pass = {key: [] for key in pass_keys.values()}
                num_active_constraints = []
                num_passed_constraints = []
                constraint_pass_rate = []

                for i in range(n):
                    metrics = {
                        "gain_dB": base_data.get("gain_dB", [None] * n)[i],
                        "power_uW": base_data.get("power_uW", [None] * n)[i],
                        "bandwidth_MHz": base_data.get("bandwidth_MHz", [None] * n)[i],
                        "bandwidth_kind": base_data.get("bandwidth_kind", [None] * n)[i],
                        "inband_flatness_db": base_data.get("inband_flatness_db", [None] * n)[i],
                        "phase_margin_status": base_data.get("phase_margin_status", [None] * n)[i],
                        "phase_margin_deg": base_data.get("phase_margin_deg", [None] * n)[i],
                        "differential_gain_db": base_data.get("differential_gain_db", [None] * n)[i],
                        "offset_v": base_data.get("offset_v", [None] * n)[i],
                        "cmrr_db": base_data.get("cmrr_db", [None] * n)[i],
                        "psrr_plus_db": base_data.get("psrr_plus_db", [None] * n)[i],
                        "psrr_minus_db": base_data.get("psrr_minus_db", [None] * n)[i],
                    }
                    goal = {
                        "gain_min": base_data.get("goal_gain_min", [None] * n)[i],
                        "power_max": base_data.get("goal_power_max", [None] * n)[i],
                        "bandwidth_min": base_data.get("goal_bandwidth_min", [None] * n)[i],
                        "phase_margin_min_deg": base_data.get("goal_phase_margin_min_deg", [None] * n)[i],
                        "phase_margin_max_deg": base_data.get("goal_phase_margin_max_deg", [None] * n)[i],
                        "offset_abs_max_v": base_data.get("goal_offset_abs_max_v", [None] * n)[i],
                        "cmrr_min_db": base_data.get("goal_cmrr_min_db", [None] * n)[i],
                        "psrr_plus_min_db": base_data.get("goal_psrr_plus_min_db", [None] * n)[i],
                        "psrr_minus_min_db": base_data.get("goal_psrr_minus_min_db", [None] * n)[i],
                        "constraint_specs": base_data.get("constraint_specs", [None] * n)[i],
                    }
                    if "goal" in base_data and isinstance(base_data["goal"][i], dict):
                        goal = base_data["goal"][i]
                    results = apply_positive_gain_gate_to_constraint_results(
                        evaluate_constraint_specs(metrics, goal),
                        metrics,
                    )
                    num_active = len(results)
                    num_passed = sum(bool(v) for v in results.values())
                    num_active_constraints.append(num_active)
                    num_passed_constraints.append(num_passed)
                    constraint_pass_rate.append((num_passed / num_active) if num_active > 0 else None)

                    for spec in spec_order:
                        derived_pass[pass_keys[spec]].append(results.get(spec))

                for key, values in derived_pass.items():
                    base_data[key] = values
                base_data["num_active_constraints"] = num_active_constraints
                base_data["num_passed_constraints"] = num_passed_constraints
                base_data["constraint_pass_rate"] = constraint_pass_rate
        except Exception:
            pass

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                # Convert numpy types to JSON-serializable Python types
                entry = tidy_dump_row(convert_to_json_serializable(entry))
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    @staticmethod
    def _normalize_split_name(split_name) -> str:
        if split_name is None:
            return "unknown"
        split = str(split_name).strip().lower()
        if not split or split == "none":
            return "unknown"
        if split == "val":
            return "validation"
        return split

    @staticmethod
    def _to_list(values):
        if values is None:
            return []
        if isinstance(values, list):
            return values
        if hasattr(values, "tolist"):
            converted = values.tolist()
            return converted if isinstance(converted, list) else [converted]
        return list(values)

    def _expand_episode_values(self, values, step_traj_uids):
        values = self._to_list(values)
        step_traj_uids = self._to_list(step_traj_uids)
        if len(values) == len(step_traj_uids):
            return values
        if not values or not step_traj_uids:
            return values

        ordered_uids = []
        seen = set()
        for uid in step_traj_uids:
            if uid not in seen:
                seen.add(uid)
                ordered_uids.append(uid)

        if len(values) != len(ordered_uids):
            return values

        uid_counts = defaultdict(int)
        for uid in step_traj_uids:
            uid_counts[uid] += 1

        expanded = []
        for value, uid in zip(values, ordered_uids):
            expanded.extend([value] * uid_counts[uid])
        return expanded

    def _extract_validation_episode_metadata(self, batch):
        episode_meta = {}
        non_tensor_batch = getattr(batch, "non_tensor_batch", {})

        if "data_source" in non_tensor_batch:
            episode_meta["data_source"] = self._to_list(non_tensor_batch["data_source"])

        if "extra_info" in non_tensor_batch:
            extra_infos = self._to_list(non_tensor_batch["extra_info"])
            episode_meta["split"] = [
                self._normalize_split_name(ei.get("split") if isinstance(ei, dict) else None) for ei in extra_infos
            ]
            episode_meta["dataset_index"] = [
                ei.get("index") if isinstance(ei, dict) else None for ei in extra_infos
            ]
        elif "split" in non_tensor_batch:
            episode_meta["split"] = [
                self._normalize_split_name(split_name) for split_name in self._to_list(non_tensor_batch["split"])
            ]

        if "env_kwargs" in non_tensor_batch:
            env_kwargs_list = self._to_list(non_tensor_batch["env_kwargs"])
            episode_meta["topology"] = [
                env_kw.get("topology", "unknown") if isinstance(env_kw, dict) else "unknown"
                for env_kw in env_kwargs_list
            ]
            episode_meta["topology_complexity"] = [
                env_kw.get("topology_complexity") if isinstance(env_kw, dict) else None
                for env_kw in env_kwargs_list
            ]
            episode_meta["signal_mode"] = [
                env_kw.get("signal_mode") if isinstance(env_kw, dict) else None
                for env_kw in env_kwargs_list
            ]
            episode_meta["goal_level"] = [
                env_kw.get("goal_level") if isinstance(env_kw, dict) else None
                for env_kw in env_kwargs_list
            ]

        if "extra_info" in non_tensor_batch:
            for target_key, source_key in (("task_id", "task_id"), ("goal_level", "goal_level"),
                                           ("signal_mode", "signal_mode"), ("topology_complexity", "complexity")):
                existing = episode_meta.get(target_key, [None] * len(extra_infos))
                if len(existing) != len(extra_infos):
                    raise ValueError("Validation task metadata must align with episode inputs.")
                episode_meta[target_key] = [
                    value if value is not None else (info.get(source_key) if isinstance(info, dict) else None)
                    for value, info in zip(existing, extra_infos)
                ]
        return episode_meta

    def _build_validation_dump_metadata(self, episode_meta, output_batch):
        extra_metadata = {}
        non_tensor_batch = getattr(output_batch, "non_tensor_batch", {})
        traj_uids = self._to_list(non_tensor_batch.get("traj_uid"))
        target_len = len(output_batch)

        for key, values in episode_meta.items():
            expanded = self._expand_episode_values(values, traj_uids)
            if len(expanded) == target_len:
                extra_metadata[key] = expanded

        turn_indices = self._to_list(non_tensor_batch.get("turn_index"))
        if len(turn_indices) == target_len:
            extra_metadata["turn_index"] = turn_indices
        if len(traj_uids) == target_len:
            extra_metadata["traj_uid"] = traj_uids
            last_idx_by_uid = {}
            for idx, uid in enumerate(traj_uids):
                last_idx_by_uid[uid] = idx
            is_final_step = [0] * target_len
            for idx in last_idx_by_uid.values():
                is_final_step[idx] = 1
            extra_metadata["is_final_step"] = is_final_step

        if "episode_lengths" in non_tensor_batch:
            episode_lengths = self._to_list(non_tensor_batch["episode_lengths"])
            if len(episode_lengths) == target_len:
                extra_metadata["episode_length"] = episode_lengths

        try:
            from agent_system.environments.env_package.circuitop.circuitop.rewards import (
                apply_positive_gain_gate_to_constraint_results,
                evaluate_constraint_specs,
            )

            step_infos = self._to_list(non_tensor_batch.get("step_info"))
            if len(step_infos) == target_len:
                spec_order = [
                    "gain",
                    "power",
                    "bandwidth",
                    "phase_margin",
                    "offset",
                    "cmrr",
                    "psrr_plus",
                    "psrr_minus",
                ]
                spec_pass_flags = {f"{spec}_pass": [] for spec in spec_order}
                num_active_constraints = []
                num_passed_constraints = []
                constraint_pass_rate = []

                for step_info in step_infos:
                    goal = step_info.get("goal") if isinstance(step_info, dict) else None
                    if not isinstance(step_info, dict) or not isinstance(goal, dict):
                        results = {}
                    else:
                        results = apply_positive_gain_gate_to_constraint_results(
                            evaluate_constraint_specs(step_info, goal),
                            step_info,
                        )

                    num_active = len(results)
                    num_passed = sum(bool(v) for v in results.values())
                    num_active_constraints.append(num_active)
                    num_passed_constraints.append(num_passed)
                    constraint_pass_rate.append((num_passed / num_active) if num_active > 0 else None)

                    for spec in spec_order:
                        spec_pass_flags[f"{spec}_pass"].append(results.get(spec))

                extra_metadata.update(spec_pass_flags)
                extra_metadata["num_active_constraints"] = num_active_constraints
                extra_metadata["num_passed_constraints"] = num_passed_constraints
                extra_metadata["constraint_pass_rate"] = constraint_pass_rate
        except Exception:
            pass

        return extra_metadata

    def _collect_validation_episode_metrics(self, output_batch, split_tags, episode_meta=None):
        try:
            from agent_system.environments.env_package.circuitop.circuitop.rewards import (
                apply_positive_gain_gate_to_constraint_results,
                check_core_success,
                evaluate_constraint_specs,
            )
        except Exception:
            apply_positive_gain_gate_to_constraint_results = None
            evaluate_constraint_specs = None
            check_core_success = None

        non_tensor_batch = getattr(output_batch, "non_tensor_batch", {})
        traj_uids = self._to_list(non_tensor_batch.get("traj_uid"))
        step_infos = self._to_list(non_tensor_batch.get("step_info"))
        split_tags = self._to_list(split_tags)

        episode_metric_dict = defaultdict(lambda: defaultdict(list))
        if not traj_uids or len(traj_uids) != len(step_infos) or len(traj_uids) != len(split_tags):
            return episode_metric_dict

        indices_by_uid = defaultdict(list)
        for idx, uid in enumerate(traj_uids):
            indices_by_uid[uid].append(idx)

        for uid_indices in indices_by_uid.values():
            idx = uid_indices[-1]  # final step index for this trajectory
            step_info_final = step_infos[idx]
            if not isinstance(step_info_final, dict):
                continue

            split_name = self._normalize_split_name(split_tags[idx])
            if check_core_success is not None:
                try:
                    success_rate = float(check_core_success(step_info_final, step_info_final.get("goal", {}) or {}))
                except Exception:
                    success_rate = float(
                        step_info_final.get(
                            "success",
                            step_info_final.get(
                                "core_success",
                                step_info_final.get("performance_success", 0),
                            ),
                        )
                    )
            else:
                success_rate = float(
                    step_info_final.get(
                        "success",
                        step_info_final.get(
                            "core_success",
                            step_info_final.get("performance_success", 0),
                        ),
                    )
                )
            performance_success_rate = float(step_info_final.get("performance_success", 0))
            performance_success_any_step = 0.0
            valid_action_values = []
            perf_score_values = []
            for traj_step_idx in uid_indices:
                step_info = step_infos[traj_step_idx]
                if not isinstance(step_info, dict):
                    continue
                valid_action_values.append(float(step_info.get("is_action_valid", 1)))
                if float(step_info.get("performance_success", 0)) > 0.0:
                    performance_success_any_step = 1.0
                try:
                    _ps = step_info.get("perf_score_post")
                    if _ps is not None:
                        perf_score_values.append(float(_ps))
                except (TypeError, ValueError):
                    pass
            valid_action_rate = float(np.mean(valid_action_values)) if valid_action_values else 1.0
            # perf_score moves long before the near-zero binary success rates do; it has insensitive zones,
            # so a flat curve does not prove "no learning".
            perf_score_final = None
            try:
                _psf = step_info_final.get("perf_score_post")
                if _psf is not None:
                    perf_score_final = float(_psf)
            except (TypeError, ValueError):
                pass

            # Grouping keys: __all__ and the task's split
            group_keys = ["__all__", split_name]

            for split_key in group_keys:
                episode_metric_dict[split_key]["success_rate"].append(success_rate)
                episode_metric_dict[split_key]["performance_success_rate"].append(performance_success_rate)
                episode_metric_dict[split_key]["performance_success_rate_any_step"].append(
                    performance_success_any_step
                )
                episode_metric_dict[split_key]["valid_action_rate"].append(valid_action_rate)
                if perf_score_final is not None:
                    episode_metric_dict[split_key]["perf_score_final"].append(perf_score_final)
                if perf_score_values:
                    episode_metric_dict[split_key]["perf_score_best_step"].append(max(perf_score_values))
                    episode_metric_dict[split_key]["perf_score_step_mean"].append(float(np.mean(perf_score_values)))

            if evaluate_constraint_specs is None:
                continue

            goal = step_info_final.get("goal") or {}
            if not isinstance(goal, dict):
                continue
            constraint_results = evaluate_constraint_specs(step_info_final, goal)
            if apply_positive_gain_gate_to_constraint_results is not None:
                constraint_results = apply_positive_gain_gate_to_constraint_results(
                    constraint_results,
                    step_info_final,
                )
            for spec, passed in constraint_results.items():
                if passed is None:
                    continue
                pass_value = float(bool(passed))
                for split_key in group_keys:
                    episode_metric_dict[split_key][f"{spec}_pass_rate"].append(pass_value)

        return episode_metric_dict

    def _validate(self, dataloader=None, val_n=None, val_batch_size=None, key_prefix="val"):
        # Defaults: self.val_dataloader, val_kwargs.n, key_prefix='val' (the live signal and best-ckpt selection).
        if dataloader is None:
            dataloader = self.val_dataloader
        if val_n is None:
            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
        if val_batch_size is None:
            val_batch_size = getattr(self, "val_batch_size", self.config.data.val_batch_size)
        reward_tensor_lst = []
        data_source_lst = []
        split_tag_lst = []
        validation_dump_extra_metadata = defaultdict(list)
        validation_episode_metric_dict = defaultdict(lambda: defaultdict(list))

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        # Collect reward extra infos for dumping to file
        reward_extra_infos_all = defaultdict(list)

        for test_data in dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            original_batch_size = len(test_batch)

            expected_val_batch_size = val_batch_size
            if expected_val_batch_size is not None and original_batch_size != expected_val_batch_size:
                print(
                    f"[VALIDATION] Skipping incomplete tail batch: "
                    f"got {original_batch_size}, expected {expected_val_batch_size}."
                )
                continue

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)
            repeated_batch_size = len(test_batch)
            validation_episode_meta = self._extract_validation_episode_metadata(test_batch)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Save env_kwargs and extra_info BEFORE pop for rollout dumps
            _saved_test_env_kwargs = test_batch.non_tensor_batch.get('env_kwargs', None)
            _saved_test_extra_info = test_batch.non_tensor_batch.get('extra_info', None)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # Copy extra_info into test_gen_batch for rollout dump propagation
            if _saved_test_extra_info is not None and 'extra_info' not in test_gen_batch.non_tensor_batch:
                test_gen_batch.non_tensor_batch['extra_info'] = _saved_test_extra_info

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            del test_batch
            test_batch = test_output_gen_batch

            if "formatted_prompt_text" in test_batch.non_tensor_batch:
                input_texts = self._to_list(test_batch.non_tensor_batch["formatted_prompt_text"])
            else:
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True)
                               for ids in test_batch.batch["prompts"]]
            if len(input_texts) != len(test_batch):
                raise ValueError("Validation prompts must align with generated turn rows.")
            sample_inputs.extend(input_texts)

            # Restore env_kwargs / extra_info for validation rollout dumps
            if _saved_test_env_kwargs is not None and 'env_kwargs' not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch['env_kwargs'] = _saved_test_env_kwargs
            if _saved_test_extra_info is not None and 'extra_info' not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch['extra_info'] = _saved_test_extra_info
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # Same reward path as training (EpisodeRewardManager on the env episode rewards); trends can differ
            # because validation samples with val_kwargs while training uses the rollout temperature.
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            # Collect extra info from reward function result
            for key, value in result.items():
                if key != "reward_tensor" and isinstance(value, (list, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        reward_extra_infos_all[key].extend(value.tolist())
                    else:
                        reward_extra_infos_all[key].extend(value)
            if "step_info" in test_batch.non_tensor_batch:
                reward_extra_infos_all["step_info"].extend(self._to_list(test_batch.non_tensor_batch["step_info"]))

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            validation_dump_metadata = self._build_validation_dump_metadata(validation_episode_meta, test_batch)
            for key, values in validation_dump_metadata.items():
                validation_dump_extra_metadata[key].extend(values)
            split_values = validation_dump_metadata.get("split")
            if split_values is None or len(split_values) != reward_tensor.shape[0]:
                split_tags = np.array(["unknown"] * reward_tensor.shape[0], dtype=object)
            else:
                split_tags = np.array(split_values, dtype=object)
            split_tag_lst.append(split_tags)
            episode_metric_dict = self._collect_validation_episode_metrics(test_batch, split_tags, episode_meta=validation_episode_meta)
            for split_name, metric_values in episode_metric_dict.items():
                for metric_name, metric_list in metric_values.items():
                    validation_episode_metric_dict[split_name][metric_name].extend(metric_list)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        
        # Dump validation generations to file if validation_data_dir is configured
        validation_data_dir = self.config.trainer.get("validation_data_dir", None)
        if validation_data_dir:
            # Convert collected extra infos to dict
            reward_extra_infos_dict = dict(reward_extra_infos_all)
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                extra_metadata=dict(validation_dump_extra_metadata),
                dump_path=validation_data_dir,
            )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        split_tags = np.concatenate(split_tag_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate split-wise score contribution (validation vs test) in a single val pass
        split_reward = {}
        for i in range(reward_tensor.shape[0]):
            split_name = split_tags[i]
            if split_name not in split_reward:
                split_reward[split_name] = []
            split_reward[split_name].append(reward_tensor[i].item())
        # Per-split keys (split_<name>/*) only when the validation set mixes several splits; with a single split they
        # would duplicate the overall keys.
        emit_split_keys = len(split_reward) > 1

        metric_dict = {}
        
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'{key_prefix}/{data_source}/test_score'] = np.mean(rewards)

        if emit_split_keys:
            for split_name, rewards in split_reward.items():
                metric_dict[f'{key_prefix}/split_{split_name}/test_score'] = np.mean(rewards)

        # Per-split episode metrics: final-step success/performance, any-step performance success, and the
        # per-trajectory mean valid-action rate.
        for split_name, metric_values in validation_episode_metric_dict.items():
            if split_name != "__all__" and not emit_split_keys:
                continue
            metric_prefix = key_prefix if split_name == "__all__" else f"{key_prefix}/split_{split_name}"
            if split_name != "__all__":
                episode_count = len(metric_values.get("success_rate", []))
                metric_dict[f"{metric_prefix}/episode_count"] = episode_count

            for metric_name, values in metric_values.items():
                if not values:
                    continue
                mean_value = float(np.mean(values))
                if metric_name.endswith("_pass_rate"):
                    spec_name = metric_name[:-len("_pass_rate")]
                    metric_key = f"{metric_prefix}/spec_pass_rate/{spec_name}"
                else:
                    metric_key = f"{metric_prefix}/{metric_name}"
                metric_dict[metric_key] = mean_value

        print(f"[VALIDATION COMPLETE]")
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # So that actor/ref workers can use the same tokenizer path when loading from a checkpoint without tokenizer:
        # inject critic.model.tokenizer_path into actor_rollout_ref.model if not set (actor config has no tokenizer_path in YAML).
        with open_dict(self.config):
            if OmegaConf.select(self.config, "actor_rollout_ref.model.tokenizer_path") is None:
                self.config.actor_rollout_ref.model.tokenizer_path = self.config.critic.model.tokenizer_path

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if os.path.isfile(os.path.join(local_global_step_folder, ".complete")):
            if (not os.path.isdir(os.path.join(local_global_step_folder, "actor"))
                    or not os.path.isfile(os.path.join(local_global_step_folder, "data.pt"))):
                raise RuntimeError(f"Completed checkpoint is missing payload: {local_global_step_folder}")
            print(f"[Checkpoint] Step {self.global_steps} already complete; not rewriting linked best weights")
            return

        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        # Write `.complete` only after actor/critic/dataloader are on disk, so readers can tell a finished checkpoint
        # from one mid-write; then update the latest pointer atomically (tmp+fsync+rename).
        with open(os.path.join(local_global_step_folder, ".complete"), "w") as f:
            f.write(str(self.global_steps))
            f.flush()
            os.fsync(f.fileno())
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        tmp_pointer = local_latest_checkpointed_iteration + ".tmp"
        with open(tmp_pointer, "w") as f:
            f.write(str(self.global_steps))
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_pointer, local_latest_checkpointed_iteration)

    def _maybe_save_best_val_checkpoint(self, val_metrics: dict):
        """Keep the top-K (trainer.max_best_ckpt_to_keep) best-val checkpoints under <default_local_dir>/best/.

        Checkpoints are hardlinked out of the rotating pool, so rotation never deletes them; tracker state persists
        to best_meta.json and is restored on resume.
        """
        import shutil

        score_key = self.config.trainer.get("val_score_key", "val/success_rate")
        if score_key:
            score = val_metrics.get(score_key)
            if score is None:
                raise RuntimeError(
                    f"[BestCkpt] val_score_key='{score_key}' not found in val_metrics. "
                    f"Available keys: {sorted(val_metrics.keys())}"
                )
            score = float(score)
        else:
            # DEPRECATED: selecting checkpoints on *test* scores is
            # test-set leakage — kept for backwards compatibility only.
            print("[BestCkpt] WARNING: empty val_score_key selects on *test_score — "
                  "DEPRECATED (test leakage); set trainer.val_score_key instead.")
            test_scores = [v for k, v in val_metrics.items() if k.endswith("/test_score")]
            if not test_scores:
                print("[BestCkpt] No *test_score keys found in val_metrics — skipping best-ckpt save.")
                return
            score = float(np.mean(test_scores))

        if not np.isfinite(score):
            raise FloatingPointError(f"Nonfinite best-checkpoint score: {score}")
        if any(step == self.global_steps for _, step in self._best_ckpt_history):
            return
        keep_k = int(self.config.trainer.get("max_best_ckpt_to_keep", 2))
        worst_kept = min((s for s, _ in self._best_ckpt_history), default=float("-inf"))
        qualifies = score > self._best_val_score or (
            len(self._best_ckpt_history) < keep_k or score > worst_kept
        )
        if not qualifies:
            print(
                f"[BestCkpt] Val score {score:.4f} did not qualify "
                f"(best {self._best_val_score:.4f} at step {self._best_val_ckpt_step}; "
                f"worst kept {worst_kept:.4f})."
            )
            return

        print(f"[BestCkpt] Val score {score:.4f} qualifies (best so far "
              f"{self._best_val_score:.4f}) at step {self.global_steps} — saving.")
        self._save_checkpoint()  # regular save into the rotating pool

        # Hardlink the checkpoint into best/ (same filesystem: no extra disk) so pool rotation cannot delete it.
        local_dir = self.config.trainer.default_local_dir
        src = os.path.join(local_dir, f"global_step_{self.global_steps}")
        best_root = os.path.join(local_dir, "best")
        dst = os.path.join(best_root, f"global_step_{self.global_steps}")
        if not os.path.isfile(os.path.join(src, ".complete")):
            raise RuntimeError(f"Best checkpoint source is incomplete: {src}")
        if not os.path.isdir(dst):
            os.makedirs(best_root, exist_ok=True)
            tmp = dst + ".tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                shutil.copytree(src, tmp, copy_function=os.link)
            except OSError:
                # cross-device fallback (should not happen: same fs)
                shutil.rmtree(tmp, ignore_errors=True)
                shutil.copytree(src, tmp)
            os.rename(tmp, dst)

        # update tracker AFTER the copy is safely in place; pointer write is
        # tmp+fsync+rename for the same reason as the latest pointer.
        if score > self._best_val_score:
            self._best_val_score = score
            self._best_val_ckpt_step = self.global_steps
            best_ptr = os.path.join(local_dir, "best_checkpointed_iteration.txt")
            with open(best_ptr + ".tmp", "w") as f:
                f.write(str(self.global_steps))
                f.flush()
                os.fsync(f.fileno())
            os.rename(best_ptr + ".tmp", best_ptr)
        self._best_ckpt_history.append((score, self.global_steps))
        self._best_ckpt_history.sort(key=lambda x: -x[0])
        for s, step in self._best_ckpt_history[keep_k:]:
            stale = os.path.join(best_root, f"global_step_{step}")
            if os.path.isdir(stale):
                shutil.rmtree(stale)
                print(f"[BestCkpt] Pruned best-pool checkpoint step {step} (score {s:.4f})")
        self._best_ckpt_history = self._best_ckpt_history[:keep_k]
        self._persist_best_ckpt_state()
        print(f"[BestCkpt] best/ now holds: "
              f"{[(round(s,4), st) for s, st in self._best_ckpt_history]}")

    def _load_checkpoint(self):
        import sys

        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            # Retry once on failure — NFS mounts can have transient hiccups.
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            if global_step_folder is None:
                import time
                time.sleep(2)
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                if global_step_folder is not None:
                    print(f"[PPO] find_latest_ckpt_path succeeded on retry: {global_step_folder}")
                    sys.stdout.flush()

        # find global_step_folder
        # Track whether we are doing a true resume (existing training checkpoint)
        # vs init-from-checkpoint (loading weights from model.path for a fresh run).
        _init_from_model_path = False

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                # Model path may point to an FSDP checkpoint (e.g. to init from checkpoint with 2 GPUs).
                # If so, load from it so we do not run with empty weights (HF init is "one" model; FSDP ckpt has 2 shards).
                model_path = getattr(self.config.actor_rollout_ref.model, "path", None)
                if model_path and os.path.isdir(model_path):
                    model_path = os.path.abspath(model_path) if not os.path.isabs(model_path) else model_path
                    actor_dir = resolve_ppo_checkpoint_model_path(model_path, "actor")
                    if os.path.isdir(actor_dir) and detect_checkpoint_format(actor_dir) == "fsdp":
                        global_step_folder = model_path
                        _init_from_model_path = True
                        print(f"[PPO] Loading initial weights from model.path (FSDP checkpoint): {global_step_folder}")
                        sys.stdout.flush()
                if global_step_folder is None:
                    print("Training from scratch")
                    sys.stdout.flush()
                    return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        if global_step_folder is None:
            print("Training from scratch")
            return 0
        print(f"Load from checkpoint folder: {global_step_folder}")
        sys.stdout.flush()
        # set global step (parse from path when it contains global_step_N; else 0 for init-from-checkpoint)
        if _init_from_model_path:
            # Fresh run from model.path: global_steps restarts at 0 unless continue_from_checkpoint_step is set or
            # model.path lies in default_local_dir (a true resume whose latest-pointer lookup failed).
            _continue_step = self.config.trainer.get("continue_from_checkpoint_step", False)
            if not _continue_step and "global_step_" in global_step_folder:
                _model_parent = os.path.dirname(os.path.abspath(global_step_folder))
                _ckpt_dir = os.path.abspath(self.config.trainer.default_local_dir)
                if _model_parent == _ckpt_dir:
                    _continue_step = True
                    print(f"[PPO] Auto-detected resume scenario (model.path parent matches default_local_dir)")
            if _continue_step and "global_step_" in global_step_folder:
                self.global_steps = int(global_step_folder.split("global_step_")[-1])
            else:
                self.global_steps = 0
        elif "global_step_" in global_step_folder:
            self.global_steps = int(global_step_folder.split("global_step_")[-1])
        else:
            self.global_steps = 0

        print(f"Setting global step to {self.global_steps}")
        sys.stdout.flush()

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # Sync loaded actor weights to ref (so ref = checkpoint policy) and to rollout (VLLM)
        if hasattr(self.actor_rollout_wg, "sync_ref_from_actor"):
            self.actor_rollout_wg.sync_ref_from_actor()
        if hasattr(self.actor_rollout_wg, "sync_rollout_weights"):
            self.actor_rollout_wg.sync_rollout_weights()
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        _skip_dataloader_state_restore = self.config.trainer.get("skip_dataloader_state_restore", False)
        if _init_from_model_path:
            print(f"[PPO] Init-from-model-path: skipping dataloader state restore (fresh training run)")
        elif _skip_dataloader_state_restore:
            print("[PPO] Config requested skipping dataloader state restore")
        else:
            dataloader_local_path = os.path.join(global_step_folder, "data.pt")
            if os.path.exists(dataloader_local_path):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                try:
                    self.train_dataloader.load_state_dict(dataloader_state_dict)
                except (StopIteration, RuntimeError) as e:
                    # A sampler state saved at epoch end raises StopIteration; weights and global_steps are already
                    # restored, so restart the epoch (some samples may repeat).
                    print(f"Warning: Failed to restore dataloader state ({type(e).__name__}: {e}). "
                          f"Continuing with fresh dataloader from epoch start.")
            else:
                print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        sys.stdout.flush()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        # Finish the trackers explicitly on every exit path of the training loop (last step, early stop,
        # plateau stop, val_only, exceptions) while the wandb service is still alive; see Tracking.finish.
        try:
            return self._fit_loop(logger)
        finally:
            logger.finish()

    def _fit_loop(self, logger):
        """Body of fit(); fit() wraps it so the trackers are always closed."""
        from omegaconf import OmegaConf  # noqa: F401  (used below)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        fit_start_time = time.time()
        progress_log_every = max(1, int(os.environ.get("TRAIN_PROGRESS_LOG_EVERY", "1")))

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            train_iter = iter(self.train_dataloader)
            while True:
                try:
                    if self._turn_batch_refill_enabled():
                        batch = self._next_turn_batch_root(train_iter)
                    else:
                        batch = DataProto.from_single_dict(next(train_iter))
                except StopIteration:
                    break
                metrics = {}
                timing_raw = {}
                turn_batch_refill_enabled = self._turn_batch_refill_enabled()
                turn_batch_refill_first_root = (
                    batch.select_idxs(np.arange(len(batch), dtype=np.int64))
                    if turn_batch_refill_enabled else None
                )

                # pop moves env_kwargs away and `batch` is replaced by the rollout output; keep env_kwargs and
                # extra_info to restore them for the rollout dumps (task_id, topology).
                _saved_env_kwargs = batch.non_tensor_batch.get('env_kwargs', None)
                _saved_extra_info = batch.non_tensor_batch.get('extra_info', None)
                current_task_preview = _format_task_preview(_saved_env_kwargs)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                # Copy extra_info into gen_batch so multi_turn_loop can propagate
                # it through step dicts for rollout dump extraction.
                if _saved_extra_info is not None and 'extra_info' not in gen_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch['extra_info'] = _saved_extra_info

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        if turn_batch_refill_enabled:
                            gen_batch_output, refill_metrics, current_task_preview = (
                                self._collect_turn_batch_refill_rollouts(
                                    first_root_batch=turn_batch_refill_first_root,
                                    first_gen_batch=gen_batch,
                                    train_iter=train_iter,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    envs=self.envs,
                                )
                            )
                            metrics.update(refill_metrics)
                            if len(gen_batch_output) == 0:
                                print(f"[SLIC] Epoch {epoch} exhausted with no informative groups; no optimizer step.")
                                continue
                        else:
                            gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                    gen_batch=gen_batch,
                                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                                    envs=self.envs,
                                                                    is_train=True,
                                                                    )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)

                    del batch
                    batch = gen_batch_output

                    # Restore env_kwargs / extra_info into the new batch so rollout
                    # dump extraction can find topology, circuit_name, task_id.
                    if _saved_env_kwargs is not None and 'env_kwargs' not in batch.non_tensor_batch:
                        batch.non_tensor_batch['env_kwargs'] = _saved_env_kwargs
                    if _saved_extra_info is not None and 'extra_info' not in batch.non_tensor_batch:
                        batch.non_tensor_batch['extra_info'] = _saved_extra_info

                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Zero the response_mask of adjust_batch filler rows so pg/entropy/KL ignore them
                    # (their singleton uid already gives GRPO advantage 0).
                    if "is_padding_sample" in batch.non_tensor_batch:
                        _pad_rows = np.asarray(batch.non_tensor_batch["is_padding_sample"]) == 1
                        if _pad_rows.any():
                            batch.batch["response_mask"][torch.as_tensor(_pad_rows)] = 0
                    # Training-health metric for episode-level GRPO: fraction of zero-signal groups (identical
                    # episode rewards within a uid group give zero advantage). SLIC logs its turn-group version after
                    # the advantage computation; MT-GRPO-tree advantages do not use episode rewards, so it is not logged.
                    _uids = batch.non_tensor_batch.get("uid")
                    _ers = batch.non_tensor_batch.get("episode_rewards")
                    if (_uids is not None and _ers is not None and self.config.algorithm.adv_estimator
                            not in (AdvantageEstimator.SLIC, AdvantageEstimator.MT_GRPO_TREE)):
                        _pad_mask = np.asarray(batch.non_tensor_batch.get(
                            "is_padding_sample", np.zeros(len(_uids), dtype=np.int32))) == 1
                        _groups = defaultdict(list)
                        for _u, _r, _p in zip(_uids, _ers, _pad_mask):
                            if not _p and _r is not None:
                                _groups[_u].append(float(_r))
                        _multi = [v for v in _groups.values() if len(v) >= 2]
                        if _multi:
                            metrics["train/degenerate_group_frac"] = float(
                                sum(1 for v in _multi if float(np.std(v)) < 1e-9) / len(_multi))
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        policy_entropy = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {
                            "actor/policy_entropy": policy_entropy.detach().item(),
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        guard = self.config.algorithm.get('rollout_logprob_guard') or {}
                        if guard.get('enabled', False):
                            from verl.utils.rollout_logprob_guard import validate_rollout_logprobs
                            if 'rollout_log_probs' not in batch.batch:
                                raise ValueError('Enabled logprob guard requires rollout_log_probs')
                            metrics.update(validate_rollout_logprobs(
                                batch.batch['rollout_log_probs'], batch.batch['old_log_probs'],
                                batch.batch['response_mask'], float(guard['mean_limit']), float(guard['p99_limit']),
                            ))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            # Preserve per-step rewards from env rollout; reward_extra_info may overwrite with wrong shape
                            n_rows = len(batch)
                            rollout_rewards = batch.non_tensor_batch.get('rewards')
                            has_rollout_rewards = (
                                rollout_rewards is not None
                                and hasattr(rollout_rewards, '__len__')
                                and len(rollout_rewards) == n_rows
                            )
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            if has_rollout_rewards and 'rewards' in reward_extra_infos_dict:
                                batch.non_tensor_batch['rewards'] = np.asarray(rollout_rewards, dtype=np.float64)

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                simulation_failure_penalty_coef=OmegaConf.select(
                                    self.config.actor_rollout_ref.actor, "simulation_failure_penalty_coef"
                                ),
                                invalid_action_penalty_mode=OmegaConf.select(
                                    self.config.actor_rollout_ref.actor, "invalid_action_penalty_mode"
                                ) or "last_token",
                            )
                            metrics.update(invalid_metrics)
                        else:
                            # Still log validity metrics even when penalty is disabled
                            metrics.update(compute_valid_action_metrics(batch))
                        
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, _ = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        slic_cfg = self.config.algorithm.get("slic", {})
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_wise_reward_key=slic_cfg.get("reward_key", "perf_score_post"),
                            step_wise_advantage_mode=slic_cfg.get("advantage_mode", "joint"),
                            step_wise_validity_weight=float(slic_cfg.get("validity_weight", 1.0)),
                            step_wise_performance_weight=float(slic_cfg.get("performance_weight", 1.0)),
                            step_wise_sepvp_weight_normalize=bool(slic_cfg.get('sepvp_weight_normalize', False)),
                            step_wise_use_fixed_invalid_turn_reward=slic_cfg.get(
                                "use_fixed_invalid_turn_reward", False
                            ),
                            step_wise_invalid_turn_reward=slic_cfg.get(
                                "invalid_turn_reward", -0.1
                            ),
                            # MT-GRPO tree: R^I = the SLIC turn reward (same reward_key / fixed failure reward as SLIC,
                            # from algorithm.slic), alpha from algorithm.mt_grpo
                            mt_grpo_alpha=float(self.config.algorithm.get("mt_grpo", {}).get("alpha", 1.0)),
                            mt_grpo_reward_key=slic_cfg.get("reward_key", "perf_score_post"),
                            mt_grpo_use_fixed_invalid_turn_reward=bool(slic_cfg.get("use_fixed_invalid_turn_reward", True)),
                            mt_grpo_invalid_turn_reward=float(slic_cfg.get("invalid_turn_reward", -0.1)),
                            mt_grpo_outcome_source=str(self.config.algorithm.get("mt_grpo", {}).get("outcome_source", "env_reward")),
                        )

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.SLIC:
                            row_signal = batch.batch["advantages"].abs().amax(dim=-1).detach().cpu().tolist()
                            active_rows = batch.batch["response_mask"].any(dim=-1).detach().cpu().tolist()
                            turn_signals = defaultdict(list)
                            for uid, turn, signal, active in zip(
                                batch.non_tensor_batch["uid"], batch.non_tensor_batch["turn_index"],
                                row_signal, active_rows,
                            ):
                                if active:
                                    turn_signals[(uid, int(turn))].append(signal)
                            if turn_signals:
                                metrics["train/degenerate_group_frac"] = float(np.mean([
                                    max(signals) < 1e-9 for signals in turn_signals.values()]))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            # Use formatted_prompt_text if available (preserves original template structure)
                            # Otherwise fall back to decoding from tokens
                            if 'formatted_prompt_text' in batch.non_tensor_batch:
                                inputs = batch.non_tensor_batch['formatted_prompt_text']
                                # Convert to list if needed
                                if not isinstance(inputs, list):
                                    inputs = inputs.tolist() if hasattr(inputs, 'tolist') else list(inputs)
                            else:
                                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            
                            # `score`: the per-turn reward the estimator trained on (SLIC/MT-GRPO: the step_wise reward, or the
                            # fixed invalid-turn reward for a failed action; otherwise the env reward). `env_reward`: the
                            # environment's own reward (0 before the done turn, the terminal bonus at it).
                            env_rewards = batch.non_tensor_batch['rewards']
                            env_rewards = env_rewards.tolist() if hasattr(env_rewards, 'tolist') else list(env_rewards)
                            n_batch = len(batch)
                            if len(env_rewards) != n_batch:
                                raise RuntimeError(f"rollout dump: {len(env_rewards)} env rewards for {n_batch} rows")
                            step_rewards = list(env_rewards)
                            adv_estimator = self.config.algorithm.adv_estimator
                            slic_cfg = self.config.algorithm.get("slic", {})
                            if adv_estimator in (AdvantageEstimator.SLIC, AdvantageEstimator.MT_GRPO_TREE):
                                step_infos_for_reward = batch.non_tensor_batch.get("step_info")
                                if step_infos_for_reward is None or len(step_infos_for_reward) != n_batch:
                                    raise RuntimeError("rollout dump: step_info missing or misaligned with the batch")
                                step_rewards = core_algos._extract_step_reward_scores(
                                    step_infos_for_reward,
                                    str(slic_cfg.get("reward_key", "perf_score_post")),
                                    dtype=torch.float32,
                                    device="cpu",
                                    use_fixed_invalid_turn_reward=bool(slic_cfg.get("use_fixed_invalid_turn_reward", False)),
                                    invalid_turn_reward=float(slic_cfg.get("invalid_turn_reward", -0.1)),
                                ).tolist()

                            extra_metadata = {'env_reward': env_rewards}
                            if adv_estimator == AdvantageEstimator.SLIC:
                                actual_mask = batch.batch['response_mask']
                                if 'loss_mask' in batch.batch:
                                    actual_mask = batch.batch['loss_mask'][:, -actual_mask.shape[-1]:]
                                for key in ('step_wise_validity_advantage', 'step_wise_performance_advantage',
                                            'step_wise_performance_eligible'):
                                    if key in batch.batch:
                                        extra_metadata[key] = batch.batch[key].detach().cpu().tolist()
                                # the row advantage (constant over the response tokens of a turn)
                                extra_metadata['step_wise_advantage'] = (
                                    (batch.batch['advantages'] * actual_mask).sum(-1) / actual_mask.sum(-1).clamp_min(1)
                                ).detach().cpu().tolist()
                            # Group structure: without these the dumps cannot recover the advantage groups.
                            for _group_key in ('uid', 'traj_uid', 'turn_index', 'slic_continuation_selected'):
                                if _group_key in batch.non_tensor_batch:
                                    extra_metadata[_group_key] = batch.non_tensor_batch[_group_key]
                            if adv_estimator == AdvantageEstimator.MT_GRPO_TREE:
                                # tree structure and the node advantage, so the tree and A can be audited
                                for _k in ('mtg_node_id', 'mtg_parent_id', 'mtg_depth', 'mtg_is_terminal'):
                                    extra_metadata[_k] = batch.non_tensor_batch[_k]
                                # R^O as the estimator used it: the terminal turn's own reward (outcome_source turn_reward) or
                                # the environment's terminal reward (env_reward); NaN on non-terminal rows
                                _terminal = np.asarray(batch.non_tensor_batch['mtg_is_terminal'], dtype=bool)
                                if str(self.config.algorithm.get("mt_grpo", {}).get("outcome_source", "env_reward")) == "turn_reward":
                                    _outcome = np.where(_terminal, np.asarray(step_rewards, dtype=np.float64), np.nan)
                                else:
                                    _outcome = np.asarray(batch.non_tensor_batch['mtg_outcome_reward'], dtype=np.float64)
                                extra_metadata['mtg_outcome_reward'] = _outcome.tolist()
                                extra_metadata['mtg_advantage'] = batch.batch['advantages'][:, 0].detach().cpu().tolist()
                            if hasattr(batch, 'non_tensor_batch'):
                                # Extract topology from env_kwargs if available
                                if 'env_kwargs' in batch.non_tensor_batch:
                                    topologies = []
                                    # Index explicitly: DataProto has no __iter__, so `for item in batch` never stops.
                                    for i in range(len(batch)):
                                        item = batch[i]
                                        env_kwargs = item.non_tensor_batch.get('env_kwargs', {})
                                        topologies.append(env_kwargs.get('topology', 'unknown'))
                                    extra_metadata['topology'] = topologies
                                    # Extract task_id from extra_info for dataset augmentation matching
                                    if 'extra_info' in batch.non_tensor_batch:
                                        task_ids = []
                                        for i in range(len(batch)):
                                            ei = batch.non_tensor_batch['extra_info'][i]
                                            task_ids.append(ei.get('task_id', '') if isinstance(ei, dict) else '')
                                        extra_metadata['task_id'] = task_ids
                                
                                # Include data_source if available
                                if 'data_source' in batch.non_tensor_batch:
                                    extra_metadata['data_source'] = batch.non_tensor_batch['data_source']
                                
                                if 'episode_lengths' in batch.non_tensor_batch:
                                    extra_metadata['episode_length'] = batch.non_tensor_batch['episode_lengths']
                                
                                # Extract info dict fields from environment step
                                if 'step_info' in batch.non_tensor_batch:
                                    step_infos = batch.non_tensor_batch['step_info']
                                    # Extract specific fields from step_info dicts
                                    if len(step_infos) > 0 and isinstance(step_infos[0], dict):
                                        extra_metadata['gain_dB'] = [info.get('gain_dB', 0.0) for info in step_infos]
                                        extra_metadata['power_uW'] = [info.get('power_uW', 0.0) for info in step_infos]
                                        extra_metadata['bandwidth_MHz'] = [info.get('bandwidth_MHz', 0.0) for info in step_infos]
                                        # "goal" (the env goal with the preset tolerances) is read by the per-spec pass derivation below and is
                                        # not written to the dump (verl/utils/rollout_dump_fields.py keeps only the flat goal_* fields).
                                        for field in ("bandwidth_kind", "inband_flatness_db", "phase_margin_status", "error_type", "goal"):
                                            extra_metadata[field] = [info.get(field) for info in step_infos]
                                        extra_metadata['phase_margin_deg'] = [info.get('phase_margin_deg') for info in step_infos]
                                        extra_metadata['offset_v'] = [info.get('offset_v') for info in step_infos]
                                        extra_metadata['cmrr_db'] = [info.get('cmrr_db') for info in step_infos]
                                        extra_metadata['psrr_plus_db'] = [info.get('psrr_plus_db') for info in step_infos]
                                        extra_metadata['psrr_minus_db'] = [info.get('psrr_minus_db') for info in step_infos]
                                        extra_metadata['perf_score_post'] = [info.get('perf_score_post', 0.0) for info in step_infos]
                                        extra_metadata['initial_gain_dB'] = [info.get('initial_gain_dB', 0.0) for info in step_infos]
                                        extra_metadata['initial_power_uW'] = [info.get('initial_power_uW', 0.0) for info in step_infos]
                                        extra_metadata['is_action_valid'] = [info.get('is_action_valid', 1) for info in step_infos]
                                        extra_metadata['error'] = [info.get('error', '') for info in step_infos]
                                        # perf_score_pre is needed to understand reward assignment
                                        extra_metadata['perf_score_pre'] = [info.get('perf_score_pre', 0.0) for info in step_infos]
                                        extra_metadata['done'] = [bool(info.get('done', False)) for info in step_infos]
                                        extra_metadata['performance_success'] = [info.get('performance_success', False) for info in step_infos]
                                        extra_metadata['goal_gain_min'] = [(info.get('goal') or {}).get('gain_min') for info in step_infos]
                                        extra_metadata['goal_power_max'] = [(info.get('goal') or {}).get('power_max') for info in step_infos]
                                        extra_metadata['goal_bandwidth_min'] = [(info.get('goal') or {}).get('bandwidth_min') for info in step_infos]
                                        extra_metadata['goal_phase_margin_min_deg'] = [(info.get('goal') or {}).get('phase_margin_min_deg') for info in step_infos]
                                        extra_metadata['goal_phase_margin_max_deg'] = [(info.get('goal') or {}).get('phase_margin_max_deg') for info in step_infos]
                                        extra_metadata['constraint_specs'] = [(info.get('goal') or {}).get('constraint_specs') for info in step_infos]
                                        extra_metadata['resulting_params'] = [info.get('resulting_params', {}) for info in step_infos]


                            # episode_reward: the environment's episode total, broadcast to every row of the trajectory
                            if 'traj_uid' in batch.non_tensor_batch:
                                traj_uids = batch.non_tensor_batch['traj_uid']
                                n_rows = len(traj_uids)
                                is_padding = np.asarray(batch.non_tensor_batch.get('is_padding_sample', np.zeros(n_rows, dtype=np.int32)))
                                if 'episode_rewards' in batch.non_tensor_batch:
                                    episode_rewards_broadcast = batch.non_tensor_batch['episode_rewards']
                                    episode_total_rewards = {}
                                    for i in range(n_rows):
                                        if is_padding[i] != 1 and traj_uids[i] not in episode_total_rewards:
                                            episode_total_rewards[traj_uids[i]] = float(episode_rewards_broadcast[i])
                                    extra_metadata['episode_reward'] = [episode_total_rewards.get(uid, 0.0) for uid in traj_uids]

                                extra_metadata['is_padding_sample'] = is_padding.tolist() if hasattr(is_padding, 'tolist') else list(is_padding)

                                # per-row episode identity for offline grouping
                                # (without it, rollouts of the same task are indistinguishable).
                                extra_metadata['traj_uid'] = [str(u) for u in traj_uids]
                                if 'turn_index' in batch.non_tensor_batch:
                                    from verl.utils.circuit_step import canonical_dump_turn_indices

                                    extra_metadata['turn_idx'] = canonical_dump_turn_indices(
                                        batch.non_tensor_batch['turn_index'], len(traj_uids)
                                    )
                                else:
                                    _tu_cnt = {}
                                    _turn_idx = []
                                    for _u in traj_uids:
                                        _turn_idx.append(_tu_cnt.get(_u, 0))
                                        _tu_cnt[_u] = _tu_cnt.get(_u, 0) + 1
                                    extra_metadata['turn_idx'] = _turn_idx

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=step_rewards,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                extra_metadata=extra_metadata,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        if self.config.trainer.get("save_best_val_checkpoint", False):
                            with _timer("save_checkpoint", timing_raw):
                                self._maybe_save_best_val_checkpoint(val_metrics)



                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))


                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if (
                    self.global_steps % progress_log_every == 0
                    or is_last_step
                    or self.global_steps == 1
                ):
                    elapsed_s = time.time() - fit_start_time
                    completed_steps = self.global_steps
                    remaining_steps = max(self.total_training_steps - completed_steps, 0)
                    avg_step_s = elapsed_s / max(completed_steps, 1)
                    eta_s = remaining_steps * avg_step_s
                    print(
                        f"[Train Progress] step={completed_steps}/{self.total_training_steps} "
                        f"{_format_progress_bar(completed_steps, self.total_training_steps)} "
                        f"epoch={epoch + 1}/{self.config.trainer.total_epochs} "
                        f"elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)} "
                        f"{_format_training_metric_summary(metrics)}"
                    )

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    progress_bar.close()
                    return
            # Save at epoch end unless save_freq just saved.
            if self.config.trainer.save_freq > 0:
                last_saved_step = self.global_steps - 1  # global_steps already incremented
                if last_saved_step % self.config.trainer.save_freq != 0:
                    print(f"[End-of-epoch] Saving checkpoint at step {last_saved_step}")
                    # Temporarily rewind so checkpoint is named correctly
                    self.global_steps = last_saved_step
                    self._save_checkpoint()
                    self.global_steps = last_saved_step + 1
