# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    raw_returns = scores.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        advantages = scores.unsqueeze(-1) * response_mask
        returns = raw_returns.unsqueeze(-1) * response_mask

    return advantages, returns


def _normalize_group_relative_scores(
    raw_scores: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
) -> torch.Tensor:
    """Normalize scalar scores within GRPO-style groups."""
    adv_scores = raw_scores.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = raw_scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(raw_scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, dtype=raw_scores.dtype, device=raw_scores.device)
                id2std[idx] = torch.tensor(1.0, dtype=raw_scores.dtype, device=raw_scores.device)
            elif len(id2score[idx]) > 1:
                group_scores = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(group_scores)
                id2std[idx] = torch.std(group_scores)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                adv_scores[i] = (raw_scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                adv_scores[i] = raw_scores[i] - id2mean[index[i]]
    return adv_scores


def _extract_step_reward_scores(
    step_info: np.ndarray,
    reward_key: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_fixed_invalid_turn_reward: bool = False,
    invalid_turn_reward: float = -0.5,
) -> torch.Tensor:
    """Extract a scalar per-row reward from step_info."""
    from verl.utils.circuit_step import uses_fixed_failure_reward

    info_arr = np.asarray(step_info, dtype=object)
    reward_values = []
    for row_idx, row_info in enumerate(info_arr):
        if not isinstance(row_info, dict):
                raise RuntimeError(
                    f"Expected dict step_info rows, got {type(row_info)} at row {row_idx}."
                )
        if use_fixed_invalid_turn_reward and uses_fixed_failure_reward(row_info):
            reward_values.append(float(invalid_turn_reward))
            continue
        if reward_key == "perf_score_delta":
            raw_post = row_info.get("perf_score_post")
            raw_pre = row_info.get("perf_score_pre")
            if raw_post is None or raw_pre is None:
                raise RuntimeError(
                    f"reward_key='perf_score_delta' requires perf_score_post and perf_score_pre at row {row_idx}."
                )
            try:
                reward_val = float(raw_post) - float(raw_pre)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid perf score delta inputs at row {row_idx}: post={raw_post!r}, pre={raw_pre!r}."
                ) from exc
        else:
            raw_val = row_info.get(reward_key)
            if raw_val is None:
                raise RuntimeError(
                    f"step_info row {row_idx} is missing reward_key='{reward_key}'."
                )
            try:
                reward_val = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid reward value for key '{reward_key}' at row {row_idx}: {raw_val!r}."
                ) from exc
        reward_values.append(reward_val)
    scores = torch.as_tensor(reward_values, dtype=dtype, device=device)
    if not torch.isfinite(scores).all():
        bad_rows = torch.nonzero(~torch.isfinite(scores), as_tuple=False).flatten().tolist()
        raise FloatingPointError(
            f"Non-finite step rewards for '{reward_key}' at rows {bad_rows}; "
            "check simulator results before computing turn advantages."
        )
    return scores


def compute_slic_group_keep_mask(
    index: np.ndarray,
    turn_index: np.ndarray,
    traj_index: np.ndarray,
    step_info: np.ndarray,
    group_size: int,
    reward_key: str = "perf_score_post",
    use_fixed_invalid_turn_reward: bool = False,
    invalid_turn_reward: float = -0.1,
    reward_tolerance: float = 0.0,
    is_padding_sample: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Keep complete same-state turn groups whose actual turn-level rewards differ."""
    size = len(index)
    if group_size < 2 or not np.isfinite(reward_tolerance) or reward_tolerance < 0:
        raise ValueError("Turn-level group filtering requires group_size >= 2 and finite nonnegative tolerance.")
    if any(len(values) != size for values in (turn_index, traj_index, step_info)):
        raise ValueError("Turn-level group filter metadata must have identical lengths.")
    padding = np.zeros(size, dtype=bool) if is_padding_sample is None else np.asarray(is_padding_sample, dtype=bool)
    if padding.shape != (size,):
        raise ValueError("Turn-level group filter padding mask must match row count.")
    active = np.flatnonzero(~padding)
    keep = np.zeros(size, dtype=bool)
    scores = _extract_step_reward_scores(
        np.asarray(step_info, dtype=object)[active], reward_key, dtype=torch.float32, device="cpu",
        use_fixed_invalid_turn_reward=use_fixed_invalid_turn_reward, invalid_turn_reward=invalid_turn_reward,
    ).numpy()
    groups = defaultdict(list)
    for position, row_index in enumerate(active):
        turn = int(turn_index[row_index])
        if turn < 0 or turn != turn_index[row_index]:
            raise ValueError("Turn-level group filter requires nonnegative integer turn indices.")
        groups[(index[row_index], turn)].append(position)
    skipped = 0
    for key, positions in groups.items():
        rows = active[positions]
        if len(rows) != group_size or len({traj_index[row_index] for row_index in rows}) != group_size:
            raise ValueError(f"Turn-level group {key!r} must contain {group_size} distinct candidate trajectories.")
        values = scores[positions]
        if float(values.max() - values.min()) <= reward_tolerance:
            skipped += 1
        else:
            keep[rows] = True
    return keep, dict(groups=len(groups), skipped_groups=skipped, kept_groups=len(groups) - skipped,
                     skipped_rows=skipped * group_size, padding_rows=int(padding.sum()))


def compute_turn_group_joint_advantage(
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    turn_index: np.ndarray,
    step_info: np.ndarray,
    reward_key: str = "perf_score_post",
    use_fixed_invalid_turn_reward: bool = False,
    invalid_turn_reward: float = -0.5,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn-level GRPO using state reward at each turn."""
    score_dtype = response_mask.dtype if torch.is_floating_point(response_mask) else torch.float32
    raw_scores = _extract_step_reward_scores(
        step_info,
        reward_key,
        dtype=score_dtype,
        device=response_mask.device,
        use_fixed_invalid_turn_reward=use_fixed_invalid_turn_reward,
        invalid_turn_reward=invalid_turn_reward,
    )
    turn_group_index = np.asarray(
        [f"{uid}::turn_{int(turn)}" for uid, turn in zip(index, turn_index)],
        dtype=object,
    )
    adv_scores = _normalize_group_relative_scores(
        raw_scores=raw_scores,
        index=turn_group_index,
        traj_index=traj_index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        compute_mean_std_cross_steps=False,
    )
    advantages = adv_scores.unsqueeze(-1) * response_mask
    returns = raw_scores.unsqueeze(-1) * response_mask
    return advantages, returns


def compute_turn_separate_validity_perf_advantage(
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    turn_index: np.ndarray,
    step_info: np.ndarray,
    reward_key: str = "perf_score_post",
    validity_weight: float = 1.0,
    performance_weight: float = 1.0,
    use_fixed_invalid_turn_reward: bool = True,
    invalid_turn_reward: float = -0.1,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    is_padding_sample: Optional[np.ndarray] = None,
    weight_normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Turn-level advantage with separately normalized validity and performance channels (SepVP).

    Within each turn group (same uid and turn), validity (1 = measurable design, 0 = failed) is normalized over all
    siblings and the turn reward over the measurable siblings only; the advantage is
    validity_weight * A_validity + performance_weight * A_performance, divided by the weight sum if ``weight_normalize``.
    ``is_padding_sample`` (adjust_batch filler rows, zero-advantage singleton groups) is only length-checked.
    """
    from verl.utils.circuit_step import uses_fixed_failure_reward

    if not use_fixed_invalid_turn_reward:
        raise ValueError("Separate turn-level normalization requires fixed failure handling for diagnostic returns.")
    weights = (validity_weight, performance_weight)
    if any(not np.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Separate turn-level advantage weights must be finite, nonnegative, and not both zero.")
    size = len(response_mask)
    if any(len(values) != size for values in (index, traj_index, turn_index, step_info)):
        raise ValueError("Separate turn-level advantage metadata must match the response batch.")
    padding = np.zeros(size, dtype=bool) if is_padding_sample is None else np.asarray(is_padding_sample).astype(bool)
    if len(padding) != size:
        raise ValueError("Separate turn-level advantage padding mask must match the response batch.")
    eligible_values = []
    for row_index, info in enumerate(step_info):
        if not isinstance(info, dict) or info.get("is_action_valid") not in (0, 1):
            raise ValueError(f"Missing or invalid action validity at row {row_index}.")
        failed = uses_fixed_failure_reward(info)
        if info.get("is_action_valid") == 1 and info.get("simulation_measurement_valid") is False:
            raise ValueError(f"Contradictory measurement validity at row {row_index}.")
        if not failed and info.get("is_action_valid") == 0 and info.get("simulation_measurement_valid") is not True:
            raise ValueError(f"Unclassified invalid action at row {row_index}.")
        eligible_values.append(not failed)
    eligible = torch.as_tensor(eligible_values, dtype=torch.bool, device=response_mask.device)
    scores = _extract_step_reward_scores(
        step_info, reward_key, dtype=torch.float32, device=response_mask.device,
        use_fixed_invalid_turn_reward=True, invalid_turn_reward=invalid_turn_reward,
    )
    validity_advantage = torch.zeros_like(scores)
    performance_advantage = torch.zeros_like(scores)
    groups = defaultdict(dict)
    for row_index, (uid, turn, trajectory) in enumerate(zip(index, turn_index, traj_index)):
        siblings = groups[(str(uid), int(turn))]
        duplicate_rows = siblings.setdefault(str(trajectory), [])
        if duplicate_rows:
            first = duplicate_rows[0]
            if eligible_values[first] != eligible_values[row_index] or scores[first] != scores[row_index]:
                raise ValueError(f"Conflicting duplicate turn-level candidate at row {row_index}.")
        duplicate_rows.append(row_index)

    def centered(values):
        if len(values) < 2 or torch.all(values == values[0]):
            return torch.zeros_like(values)
        result = values - values.mean()
        if norm_adv_by_std_in_grpo:
            result = result / (values.std() + epsilon)
        return result

    for siblings in groups.values():
        candidate_rows = list(siblings.values())
        representatives = [rows[0] for rows in candidate_rows]
        group_validity = centered(eligible[representatives].to(scores.dtype))
        measurable_positions = [position for position, row in enumerate(representatives) if eligible_values[row]]
        measurable_rows = [representatives[position] for position in measurable_positions]
        group_performance = centered(scores[measurable_rows])
        for position, rows in enumerate(candidate_rows):
            validity_advantage[rows] = group_validity[position]
        for quality_position, position in enumerate(measurable_positions):
            performance_advantage[candidate_rows[position]] = group_performance[quality_position]
    combined = validity_weight * validity_advantage + performance_weight * performance_advantage
    if weight_normalize:
        combined = combined / (validity_weight + performance_weight)
    advantages = combined.unsqueeze(-1) * response_mask
    returns = scores.unsqueeze(-1) * response_mask
    components = {
        "step_wise_validity_advantage": validity_advantage,
        "step_wise_performance_advantage": performance_advantage,
        "step_wise_performance_eligible": eligible,
    }
    return advantages, returns, components


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        # Rows without loss-mask tokens (mask_pad filler, empty responses) are excluded: their 0/0 = nan would make
        # dp_actor skip the whole step. A micro-batch with no valid row contributes 0 (graph kept).
        seq_tokens = torch.sum(loss_mask, dim=-1)
        valid = seq_tokens > 0
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / seq_tokens.clamp(min=1)  # token-mean per valid row
        loss = seq_losses[valid].mean() if bool(valid.any()) else seq_losses.sum() * 0.0  # seq-mean over valid rows
    elif loss_agg_mode == "minibatch-token-mean":
        # Token sum; dp_actor.update_policy divides by the mini-batch token count, giving the mini-batch token-mean
        # (DAPO / Dr. GRPO: per-token weight A, not A/L) however it is split into micro-batches. Plain "token-mean"
        # with one response per micro-batch would reduce to seq-mean-token-mean. Only dp_actor.update_policy uses it.
        loss = torch.sum(loss_mat * loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def _mt_grpo_zscore(values: np.ndarray, epsilon: float) -> np.ndarray:
    """z-score of a 1-D group; a group of size < 2 or with zero spread carries no signal (all 0)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return np.zeros_like(values)
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std <= epsilon:
        return np.zeros_like(values)
    return (values - float(values.mean())) / (std + epsilon)


def compute_mt_grpo_tree_node_advantages(
    task_index: np.ndarray,
    node_id: np.ndarray,
    parent_id: np.ndarray,
    is_terminal: np.ndarray,
    turn_reward: np.ndarray,
    outcome_reward: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """MT-GRPO (arXiv 2505.11821 v3) node advantages on a full rollout tree.

    One row per tree node (= one generated turn). Siblings (rows with the same parent) were sampled from the SAME env state.
      A^I(node) = z-score of the turn reward R^I among its siblings (group of 1 / zero spread -> 0)
      A^O(leaf) = z-score of the episode outcome R^O over all terminal nodes (= trajectories) of the task
      A(node)   = A^O(node)                                      if the node ends its episode (done, or the last turn)
                = A^I(node) + alpha * mean_{children c} A(c)     otherwise
    The recursion equals the paper's per-trajectory A_k = sum_{l=k}^{K-1} alpha^(l-k) A^I_l + alpha^(K-k) A^O averaged
    with uniform weight per branch over the node's descendant trajectories. A non-terminal node without children is an error.
    Returns (advantage per row, components dict with a_turn / a_outcome / depth per row)."""
    n = len(node_id)
    task_index = np.asarray(task_index, dtype=object)
    node_id = np.asarray(node_id, dtype=object)
    parent_id = np.asarray(parent_id, dtype=object)
    is_terminal = np.asarray(is_terminal, dtype=bool)
    turn_reward = np.asarray(turn_reward, dtype=np.float64)
    outcome_reward = np.asarray(outcome_reward, dtype=np.float64)
    if not np.isfinite(turn_reward).all():
        raise RuntimeError("MT-GRPO: non-finite turn reward in the batch.")
    if not np.isfinite(outcome_reward[is_terminal]).all():
        raise RuntimeError("MT-GRPO: terminal node without a finite outcome reward.")
    pos = {}
    for i, nid in enumerate(node_id):
        if nid in pos:
            raise RuntimeError(f"MT-GRPO: duplicate node id {nid!r}.")
        pos[nid] = i
    a_turn = np.zeros(n, dtype=np.float64)
    a_out = np.full(n, np.nan, dtype=np.float64)
    # A^I: sibling groups = same (task, parent)
    groups: Dict[Tuple[Any, Any], List[int]] = defaultdict(list)
    for i in range(n):
        groups[(task_index[i], parent_id[i])].append(i)
    for rows in groups.values():
        a_turn[rows] = _mt_grpo_zscore(turn_reward[rows], epsilon)
    # A^O: terminal nodes of each task
    task_rows: Dict[Any, List[int]] = defaultdict(list)
    for i in np.flatnonzero(is_terminal):
        task_rows[task_index[i]].append(int(i))
    for rows in task_rows.values():
        a_out[rows] = _mt_grpo_zscore(outcome_reward[rows], epsilon)
    children: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        if parent_id[i] in pos:
            children[parent_id[i]].append(i)
    adv = np.full(n, np.nan, dtype=np.float64)

    def solve(i: int) -> float:
        if not np.isnan(adv[i]):
            return adv[i]
        if is_terminal[i]:
            adv[i] = a_out[i]
            return adv[i]
        kids = children.get(node_id[i], [])
        if not kids:
            raise RuntimeError(f"MT-GRPO: non-terminal node {node_id[i]!r} has no children in the batch.")
        adv[i] = a_turn[i] + float(alpha) * float(np.mean([solve(c) for c in kids]))
        return adv[i]

    # deepest nodes first, so solve() only ever recurses into already-solved children (no deep recursion)
    depth = np.zeros(n, dtype=np.int64)
    for i in range(n):
        d, p = 0, parent_id[i]
        while p in pos:
            d += 1
            p = parent_id[pos[p]]
        depth[i] = d
    for i in sorted(range(n), key=lambda i: -depth[i]):
        solve(i)
    if not np.isfinite(adv).all():
        raise RuntimeError("MT-GRPO: non-finite node advantage.")
    return adv, {"a_turn": a_turn, "a_outcome": a_out, "depth": depth}


def compute_mt_grpo_tree_advantage(
    response_mask: torch.Tensor,
    index: np.ndarray,
    node_id: np.ndarray,
    parent_id: np.ndarray,
    is_terminal: np.ndarray,
    outcome_reward: np.ndarray,
    step_info: np.ndarray,
    reward_key: str = "perf_score_post",
    use_fixed_invalid_turn_reward: bool = True,
    invalid_turn_reward: float = -0.1,
    alpha: float = 1.0,
    epsilon: float = 1e-6,
    outcome_source: str = "env_reward",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Token-level wrapper: R^I is SLIC's turn reward from step_info (reward_key, or the fixed failure reward for invalid /
    failed turns); every response token of a turn gets its node advantage. Returns (advantages, returns), both (bs, L).

    outcome_source: where R^O of a terminal node comes from.
      "env_reward"  — the env reward of the terminal turn (`outcome_reward`). Default.
      "turn_reward" — the terminal turn's own R^I."""
    dtype = response_mask.dtype if torch.is_floating_point(response_mask) else torch.float32
    turn_reward = _extract_step_reward_scores(
        step_info, reward_key, dtype=dtype, device=response_mask.device,
        use_fixed_invalid_turn_reward=use_fixed_invalid_turn_reward, invalid_turn_reward=invalid_turn_reward,
    ).detach().cpu().numpy()
    if outcome_source == "env_reward":
        outcome = np.asarray(outcome_reward, dtype=np.float64)
    elif outcome_source == "turn_reward":
        # terminal rows take their own turn reward; non-terminal rows keep NaN (never read: only terminal nodes use R^O)
        outcome = np.where(np.asarray(is_terminal, dtype=bool), turn_reward.astype(np.float64), np.nan)
    else:
        raise ValueError(f"MT-GRPO: unknown outcome_source {outcome_source!r} (expected 'env_reward' or 'turn_reward').")
    adv, _ = compute_mt_grpo_tree_node_advantages(
        task_index=index, node_id=node_id, parent_id=parent_id, is_terminal=is_terminal,
        turn_reward=turn_reward, outcome_reward=outcome, alpha=alpha, epsilon=epsilon,
    )
    adv_t = torch.as_tensor(adv, dtype=dtype, device=response_mask.device).unsqueeze(-1) * response_mask
    return adv_t, adv_t
