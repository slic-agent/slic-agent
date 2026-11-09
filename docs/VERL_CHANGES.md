# Changes to verl

`verl/` is a modified copy of [verl](https://github.com/volcengine/verl) as bundled with
[verl-agent](https://github.com/langfengQ/verl-agent) (verl version 0.3.1.dev), distributed under the Apache License 2.0
(see `LICENSE` and `Notice.txt`). Files added or modified by this project are listed below; all other files under
`verl/` are unchanged apart from identifier renaming (turn-level GRPO internals renamed to `slic` /
`step_wise`) applied to the whole tree.

## Modified (22)

- `verl/models/transformers/monkey_patch.py`
- `verl/protocol.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/sft_trainer.yaml`
- `verl/trainer/fsdp_sft_trainer.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/metric_utils.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/checkpoint/checkpoint_manager.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/utils/dataset/multiturn_sft_dataset.py`
- `verl/utils/dataset/rl_dataset.py`
- `verl/utils/dataset/sft_dataset.py`
- `verl/utils/fs.py`
- `verl/utils/fsdp_utils.py`
- `verl/utils/tracking.py`
- `verl/utils/vllm_utils.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/workers/sharding_manager/fsdp_vllm.py`

## Added (15)

- `verl/trainer/config/env_evaluation.yaml`
- `verl/trainer/main_env_eval.py`
- `verl/utils/chat_template.py`
- `verl/utils/circuit_checkpoint.py`
- `verl/utils/circuit_step.py`
- `verl/utils/dataset/dataset_utils.py`
- `verl/utils/dataset/stratified_val_sampler.py`
- `verl/utils/episode_metrics.py`
- `verl/utils/fsdp_checkpoint_loader.py`
- `verl/utils/ray_resource_check.py`
- `verl/utils/rollout_dump_fields.py`
- `verl/utils/rollout_logprob_guard.py`
- `verl/utils/slic_groups.py`
- `verl/utils/vllm_engine_config.py`
- `verl/workers/rollout/sglang_rollout/README.md`
