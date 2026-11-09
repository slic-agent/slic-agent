#!/bin/bash
# Evaluation with the paper's protocol: 8 turns, history 1, n = 3 samples per task at T = 0.7 / top-p 0.9, Pass@1 and
# Pass@3, four-spec success under the released tolerances, 65 nm, 140 parallel environments, ngspice.
# `slic eval --config configs/eval/<set>.yaml` runs this script with the checkpoint, task set and run name of the config
# and its overrides appended; "$@" comes last and wins (e.g. env.circuitop.tech_node=22nm, rollout.n=1).
# Outputs: outputs/eval/<evaluation.experiment_name>/{results.json, rollouts/, sim_workdir/}.
cd "$(dirname "$0")/../.." || exit 1
python3 -m verl.trainer.main_env_eval \
    model.path=models/my-slic-random \
    data.val_files=data/standard/test_iid.parquet \
    data.val_batch_size=140 \
    data.max_prompt_length=13072 \
    data.max_response_length=2048 \
    data.truncation=middle \
    data.trust_remote_code=true \
    env.seed=42 \
    env.max_steps=8 \
    env.history_length=1 \
    env.resources_per_worker.num_cpus=0.1 \
    env.circuitop.tech_node=65nm \
    env.circuitop.reward_config=configs/reward_config_example.yaml:sparse_constraint_done_4spec_hint \
    env.circuitop.think_prefilled=true \
    env.circuitop.require_think_tags=true \
    env.circuitop.enforce_think_prefix=false \
    env.circuitop.reject_trailing_content=false \
    env.circuitop.end_on_trailing_content=false \
    env.circuitop.end_on_think_prefix_error=false \
    env.circuitop.end_on_invalid_format=false \
    rollout.name=vllm \
    rollout.n=3 \
    rollout.temperature=0.7 \
    rollout.top_p=0.9 \
    rollout.do_sample=true \
    rollout.gpu_memory_utilization=0.78 \
    rollout.max_num_seqs=512 \
    rollout.enforce_eager=true \
    rollout.free_cache_engine=true \
    "evaluation.pass_at_ks=[1,3]" \
    evaluation.experiment_name=slic_random_qwen3.5_4b_standard_iid \
    evaluation.project_name=slic-agent-eval \
    "evaluation.logger=['console']" \
    'evaluation.output_path=outputs/eval/${evaluation.experiment_name}/results.json' \
    'evaluation.rollout_data_dir=outputs/eval/${evaluation.experiment_name}/rollouts' \
    'env.circuitop.workdir=outputs/eval/${evaluation.experiment_name}/sim_workdir' \
    ray_init.num_cpus=64 \
    "$@"
