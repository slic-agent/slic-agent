# Reproducing the paper

| paper                                          | config                                                         |
|------------------------------------------------|----------------------------------------------------------------|
| SLIC-Random (main, Qwen3.5-4B)                 | `configs/train/slic_random.yaml`                               |
| SLIC-Softmax / SLIC-Greedy                     | `configs/train/slic_softmax.yaml`, `slic_greedy.yaml`          |
| SLIC-Random + dynamic filtering                | `configs/train/slic_random_filter.yaml`                        |
| separate vs joint advantage normalisation      | `configs/train/abl_separate_norm.yaml`, `abl_joint_norm.yaml`  |
| without the success-slack bonus                | `configs/train/abl_no_slack.yaml`                              |
| without validity gates and hint                | `configs/train/abl_no_gates.yaml` (evaluate with `configs/eval/*_nohint.yaml`) |
| MT-GRPO baseline (K = 4, G = 4)                | `configs/train/mt_grpo.yaml`                                   |
| Qwen3.5-9B / Qwen3-8B                          | `configs/train/slic_random_qwen3.5_9b.yaml`, `slic_random_qwen3_8b.yaml` |
| evaluation (IID / OOD / hard / technology nodes) | `configs/eval/*.yaml`                                        |

Every RL config starts from an SFT checkpoint of the base model, which is not released: build one with `configs/sft/` on
your own trajectories and set `model_path`. The paper trained with 8-turn episodes; the launcher defaults to 4, so add
`--override env.max_steps=8` to match. `configs/train/episode_grpo.yaml` is an episode-level GRPO reference that is not in
the paper.

**Data.** The released task sets are a corrected version of the ones the paper used. We have only spot-checked that results
on them are similar; with limited compute, a full re-evaluation is left to the next release.

## Checkpoints

Training writes FSDP checkpoints under `outputs/checkpoints/<experiment>/` (`best/global_step_N` and the latest). Convert
the actor to an HF-format directory for evaluation or serving:

```bash
python scripts/checkpoint/export_qwen35_checkpoint.py --src outputs/checkpoints/<experiment>/best/global_step_N \
    --base models/my-sft-qwen3.5-4b --dst models/my-slic-random      # --base: the model the run started from
```

(`scripts/checkpoint/merge_checkpoint.py --backend fsdp` for other models.)

## Randomness

Seeds fix sampling, environments and the simulator, but vLLM batching and FSDP reductions are not bit-reproducible across
hardware; expect a few points of run-to-run variation in success rate.
