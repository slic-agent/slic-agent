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

from verl import DataProto
import torch
import numpy as np


def build_episode_component_reward_extra_info(non_tensor_batch):
    """Broadcast scalar fields of each trajectory's final step_info to all of its rows, as `episode_<field>`."""
    step_infos = non_tensor_batch.get("step_info")
    traj_uids = non_tensor_batch.get("traj_uid")
    if step_infos is None or traj_uids is None:
        return {}

    step_infos = list(step_infos)
    traj_uids = list(traj_uids)
    if len(step_infos) != len(traj_uids):
        return {}

    last_idx_by_uid = {}
    for idx, uid in enumerate(traj_uids):
        last_idx_by_uid[uid] = idx

    final_scalar_fields = set()
    scalar_field_types = (bool, int, float, np.bool_, np.integer, np.floating)
    for idx in last_idx_by_uid.values():
        step_info = step_infos[idx]
        if not isinstance(step_info, dict):
            continue
        for key, value in step_info.items():
            if isinstance(value, scalar_field_types):
                final_scalar_fields.add(key)

    reward_extra_info = {}
    for field in sorted(final_scalar_fields):
        values = []
        for uid in traj_uids:
            step_info = step_infos[last_idx_by_uid[uid]]
            value = step_info.get(field) if isinstance(step_info, dict) else None
            if isinstance(value, scalar_field_types):
                values.append(float(value))
            else:
                values.append(np.nan)
        reward_extra_info[f"episode_{field}"] = values

    return reward_extra_info

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        """Place each row's episode reward (optionally length-normalized) on its last response token."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)


            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']


            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score, dtype=torch.float32, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                # No example printing. The counter and the random draw above still run: they advance the global
                # NumPy RNG stream that rollout sampling also draws from.
                already_print_data_sources[data_source] += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": build_episode_component_reward_extra_info(data.non_tensor_batch),
            }
        else:
            return reward_tensor
