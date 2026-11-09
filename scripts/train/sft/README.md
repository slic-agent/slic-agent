# Supervised warm-up (SFT)

`scripts/train/sft/run_sft.sh` fine-tunes a base model on multi-turn optimization trajectories with
`verl.trainer.fsdp_sft_trainer`; `slic train --config configs/sft/qwen3.5_4b.yaml` drives it.

Input: two parquet files (`train`, `test`) with a `messages` column holding the chat turns
`[{"role": "system"|"user"|"assistant", "content": ...}, ...]` of one episode; only assistant
turns contribute to the loss. The paper's RL runs start from an SFT model (not released) whose
trajectories were produced by a stronger teacher model solving tasks from the `standard`
training pool; that corpus is not released either, but any trajectory source with the same chat
format works (e.g. episodes recorded with `slic solve --jsonl`).

`chat_template_think_prefill.jinja` is the chat template used for the paper's checkpoints
(`chat_template:` in `configs/sft/qwen3.5_4b.yaml`): it prefills `<think>` in the generation prompt,
which is the convention of the Qwen3 / Qwen3.5 checkpoints (`think_prefilled: true` in the eval configs).
