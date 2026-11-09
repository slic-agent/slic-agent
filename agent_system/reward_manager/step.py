from verl import DataProto
import torch
import numpy as np


class StepRewardManager:
    """Reward manager that places per-step environment rewards on response tokens.

    Unlike EpisodeRewardManager which broadcasts the accumulated episode reward,
    this manager uses the raw per-step reward returned by env.step() for each
    turn.  This is required by SLIC which computes turn-level advantages
    from individual step rewards rather than cumulative episode returns.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch['data_source']

            step_reward = float(data_item.non_tensor_batch['rewards'])

            if self.normalize_by_length:
                episode_lengths = data_item.non_tensor_batch.get('episode_lengths', 1)
                score = step_reward / max(float(episode_lengths), 1)
            else:
                score = step_reward

            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                # No example printing. The counter and the random draw above still run: they advance the global
                # NumPy RNG stream that rollout sampling also draws from.
                already_print_data_sources[data_source] += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
