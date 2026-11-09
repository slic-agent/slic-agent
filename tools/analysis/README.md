# Analysis tools

Post-hoc analyses of `slic eval` runs. Every tool reads only the per-turn rollout dumps an evaluation writes
(`<output_dir>/rollouts/batch_<i>.jsonl`, one JSON row per turn, see `slic eval --help`) and takes the directories on the
command line; nothing is looked up elsewhere. Run them from the repository root with `python -m tools.analysis.<tool>`.
The fields a dump row can carry, and their order, are listed in `verl/utils/rollout_dump_fields.py` (shared by the
training, validation and evaluation writers).

| tool | what it computes | input |
|---|---|---|
| `turn_success` | success by turn, by spec combination and by topology complexity; pass@k; best figure of merit by turn; Markdown + CSV + JSON | one `rollouts/` dir (+ the task parquet for complexity metadata) |
| `eval_ci` | pass@1 at a turn cutoff with a task-level bootstrap confidence interval, and the spread of a single-sample evaluation | one `rollouts/` dir |
| `paired_bootstrap` | paired task-level bootstrap between two runs on the same task set: difference, CI, p-value | two `rollouts/` dirs |
| `passk_turns` | success within t turns (mean ± std over sample indices) and pass@k curves, one figure | labelled `rollouts/` dirs per split |
| `tolerance_sweep` | four-spec success re-judged at tolerances 0–15 % (needs a 0 %-tolerance run for exact values below the official 5 %) | labelled `rollouts/` dirs per split |
| `tech_node_sweep` | grouped bar chart of pass@1 per technology node (or any grouping) | `model=node=split=rollouts` entries |

Examples, after `slic eval --config configs/eval/standard_iid.yaml` (output under `outputs/eval/<run_tag>/`):

```bash
R=outputs/eval/slic_random_qwen3.5_4b_standard_iid/rollouts
python -m tools.analysis.turn_success --rollout-dir $R --data-path data/standard/test_iid.parquet --split test_iid \
    --output-json figs/turn_success.json --output-csv figs/turn_success.csv --output-markdown figs/turn_success.md
python -m tools.analysis.eval_ci --rollout-dir $R --rollouts-per-problem 3 --max-turn 8
python -m tools.analysis.paired_bootstrap --run-a $R --run-b outputs/eval/other_run/rollouts --turns 4 8
python -m tools.analysis.passk_turns --out figs/passk_turns --iid "SLIC-Random=$R" --ood "SLIC-Random=outputs/eval/slic_random_qwen3.5_4b_standard_ood/rollouts"
python -m tools.analysis.tolerance_sweep --out figs/tolerance_sweep \
    --iid "SLIC-Random=$R:outputs/eval/slic_random_qwen3.5_4b_standard_iid_tol0/rollouts"
python -m tools.analysis.tech_node_sweep --out figs/tech_node_sweep --run "SLIC-Random=65nm=iid=$R" --run "SFT=65nm=iid=outputs/eval/sft_iid/rollouts"
```

Notes:

* A 32-turn evaluation (`--set max_turns=32` or `max_turns: 32` in the config) gives `passk_turns` the curve to saturation;
  an evaluation with `samples_per_task: 8` (or 32) gives its pass@k panel.
* `tolerance_sweep` below the official tolerance: the standard protocol stops an episode at its first success at 5 %, so the
  standard dumps only bound success at tighter tolerances from below. `configs/eval/standard_{iid,ood}_tol0.yaml` run the same
  checkpoint with every spec tolerance at 0 % (targets stated at 0 % in the prompt, auto-stop only at exact success); pass that
  run as the second path to get exact values.
* `tech_node_sweep` at other nodes: `slic data --cards` fetches the 22 / 32 / 45 / 90 / 130 nm cards and `slic data --family nodes`
  the validation / OOD tasks rebuilt at each node; `configs/eval/node_validation.yaml` / `node_ood.yaml` run one node (set
  `tech_node` and `tasks` together; docs/DATASETS.md). The 65 nm bars are the standard `validation` / `test_ood` runs.
