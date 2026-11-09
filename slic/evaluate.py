"""`slic eval --config configs/eval/<set>.yaml` — multi-turn evaluation with a running Pass@1 readout.

scripts/eval/run_eval.sh's command line holds the paper's evaluation protocol (8 turns, n = 3 at T = 0.7, Pass@1 / Pass@3, 65 nm);
a config lists only what differs from it. Config keys and the hydra overrides they become:
  model_path            model.path — HF-format checkpoint directory or hub id (required)
  tasks                 data.val_files — task parquet, e.g. data/standard/test_iid.parquet (required)
  run_tag               evaluation.experiment_name (default <model dir>-<parquet name>)
  output_dir            where results.json, rollouts/ and sim_workdir/ go (default outputs/eval/<run_tag>)
  wandb (bool)          evaluation.logger=['console','wandb']        project_name   evaluation.project_name
  protocol: {key: value}  named protocol settings, each the override in PROTOCOL below (max_turns -> env.max_steps,
                        samples_per_task -> rollout.n, tech_node -> env.circuitop.tech_node, pass_at_ks -> evaluation.pass_at_ks, ...);
                        reward_preset -> env.circuitop.reward_config=<reward_config_example.yaml>:<preset>
  overrides: [k=v]      hydra overrides appended after those (later ones win); env: {VAR: value} runtime environment variables
                        (simulator: CIRCUIT_SIM_BACKEND, NGSPICE_BIN; docs/SIMULATORS.md)

Command line: `--set KEY=VALUE` sets a config key or a protocol key above (--set samples_per_task=1) and is a hydra override
otherwise (rollout.tensor_model_parallel_size=2); `--override k=v` is always a hydra override. Command-line overrides come
after the config's.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import yaml

from ._common import REPO_ROOT, ConfigError, as_env_value, load_yaml, merge_env, print_plan, resolve_path, stream_subprocess
from .train import RETIRED_PREFIXES, check_env

# protocol key -> hydra key of scripts/eval/run_eval.sh's command line
PROTOCOL = {
    "max_turns": "env.max_steps",
    "history_length": "env.history_length",
    "samples_per_task": "rollout.n",
    "temperature": "rollout.temperature",
    "top_p": "rollout.top_p",
    "batch_size": "data.val_batch_size",
    "max_episodes": "evaluation.max_episodes",
    "pass_at_ks": "evaluation.pass_at_ks",
    "think_prefilled": "env.circuitop.think_prefilled",
    "require_think_tags": "env.circuitop.require_think_tags",
    "max_prompt_length": "data.max_prompt_length",
    "max_response_length": "data.max_response_length",
    "tech_node": "env.circuitop.tech_node",
    "tensor_parallel": "rollout.tensor_model_parallel_size",
    "gpu_memory_utilization": "rollout.gpu_memory_utilization",
    "max_num_seqs": "rollout.max_num_seqs",
    "seed": "env.seed",
}
REWARD_CONFIG = "configs/reward_config_example.yaml"   # resolved by RewardConfig.from_yaml against the circuitop env package
PROTOCOL_KEYS = set(PROTOCOL) | {"reward_preset"}
CONFIG_KEYS = {"model_path", "tasks", "run_tag", "output_dir", "wandb", "project_name"}


def _override(key: str, value) -> str:
    return f"{key}=null" if value is None else f"{key}={as_env_value(value)}"


def default_run_tag(model_path: str, tasks: str) -> str:
    return re.sub(r"[^a-z0-9._-]", "_", Path(model_path).name.lower()) + "-" + Path(tasks).stem


def build(cfg: Dict) -> tuple[Dict[str, str], List[str], Path]:
    """Returns (environment, command, results.json path)."""
    if "model_path" not in cfg or "tasks" not in cfg:
        raise ConfigError("eval config needs model_path and tasks")
    model = resolve_path(cfg["model_path"])
    tasks = resolve_path(cfg["tasks"])
    if not os.path.isfile(tasks):
        raise ConfigError(f"missing parquet: {tasks} (run: slic data --family standard)")
    batch = (cfg.get("protocol") or {}).get("batch_size")
    if batch:   # the evaluator drops an incomplete last batch (data.drop_last_batch), which would silently skip tasks
        import pyarrow.parquet as pq
        n = pq.ParquetFile(tasks).metadata.num_rows
        if n % int(batch):
            divisors = [d for d in range(1, n + 1) if n % d == 0 and d <= 512]
            raise ConfigError(f"protocol.batch_size={batch} does not divide the {n} tasks of {tasks} (the last {n % int(batch)} would "
                              f"be skipped); use a divisor, e.g. {divisors[-4:]}")
    run_tag = cfg.get("run_tag") or default_run_tag(model, tasks)
    out_dir = Path(resolve_path(cfg["output_dir"])) if cfg.get("output_dir") else REPO_ROOT / "outputs" / "eval" / run_tag
    ov = [f"model.path={model}", f"data.val_files={tasks}", f"evaluation.experiment_name={run_tag}"]
    if cfg.get("output_dir"):
        ov += [f"evaluation.output_path={out_dir}/results.json", f"evaluation.rollout_data_dir={out_dir}/rollouts",
               f"env.circuitop.workdir={out_dir}/sim_workdir"]
    if cfg.get("wandb"):
        ov.append("evaluation.logger=['console','wandb']")
    if cfg.get("project_name"):
        ov.append(f"evaluation.project_name={cfg['project_name']}")
    proto = cfg.get("protocol") or {}
    unknown = sorted(set(proto) - PROTOCOL_KEYS)
    if unknown:
        raise ConfigError(f"protocol: unknown key(s) {', '.join(unknown)}; known: {', '.join(sorted(PROTOCOL_KEYS))}")
    for key, hydra_key in PROTOCOL.items():
        if key in proto:
            ov.append(_override(hydra_key, proto[key]))
    if "reward_preset" in proto:
        ov.append(f"env.circuitop.reward_config={REWARD_CONFIG}:{proto['reward_preset']}")
    env = merge_env(cfg.get("env"))
    check_env(env, "config env:")
    cmd = ["bash", str(REPO_ROOT / "scripts" / "eval" / "run_eval.sh")] + ov + [str(o) for o in cfg.get("overrides") or []]
    return env, cmd, out_dir / "results.json"


def split_sets(sets: List[str], cfg: Dict) -> List[str]:
    """--set KEY=VALUE: a config or protocol key updates cfg (typed like the YAML); anything else is a hydra override."""
    overrides: List[str] = []
    for item in sets:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if k in CONFIG_KEYS:
            cfg[k] = yaml.safe_load(v) if v else None
        elif k in PROTOCOL_KEYS:
            cfg.setdefault("protocol", {})[k] = yaml.safe_load(v) if v else None   # samples_per_task=1, pass_at_ks=[1]
        elif k.startswith(RETIRED_PREFIXES):
            check_env({k: v}, "--set")
        else:
            overrides.append(item)
    return overrides


class RunningPassAt1:
    """Parse the evaluator's per-batch result blocks and keep a running four-spec success rate (Pass@1)."""

    _batch = re.compile(r"^\[Batch (\d+)/(\d+)\] Results:")
    _metric = re.compile(r"^\s{2}(\w+): ([-+0-9.eE]+)$")

    def __init__(self) -> None:
        self.in_block = False
        self.batches = 0
        self.total = 0
        self.sum_success = 0.0

    def __call__(self, line: str) -> None:
        m = self._batch.match(line)
        if m:
            self.in_block = True
            self.batches, self.total = int(m.group(1)), int(m.group(2))
            self._got: Dict[str, float] = {}
            return
        if not self.in_block:
            return
        m = self._metric.match(line)
        if m:
            self._got[m.group(1)] = float(m.group(2))
            return
        # first non-metric line closes the block
        self.in_block = False
        if "success_rate" in self._got:
            self.sum_success += self._got["success_rate"]
            n = self.batches
            print(f"[slic] running Pass@1 after batch {n}/{self.total}: success={self.sum_success / n:.4f}"
                  f"  (this batch {self._got['success_rate']:.4f})", flush=True)


def summarize(results_file: Path) -> None:
    if not results_file.exists():
        print(f"[slic] no results file at {results_file}")
        return
    data = json.loads(results_file.read_text())
    metrics = data.get("metrics", {})
    print("[slic] final results")
    print(f"  tasks={data.get('total_episodes')} rollouts={data.get('total_rollouts')}")
    for key in ("success_rate", "performance_success_rate", "validity_rate"):
        if key in metrics:
            v = metrics[key]
            v = v.get("mean", v) if isinstance(v, dict) else v
            if isinstance(v, list):   # results.json stores one value per evaluation batch; the run reports their mean
                if not v:
                    raise ValueError(f"{key}: empty list in results.json")
                v = sum(float(x) for x in v) / len(v)
            print(f"  {key}: {float(v):.4f}")
    for k, v in sorted((data.get("pass_at_k") or {}).items()):
        print(f"  {k}: {v}")
    print(f"  file: {results_file}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", help="override model_path")
    ap.add_argument("--tasks", help="override the task parquet")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="a config or protocol key (samples_per_task=1) or else a hydra override")
    ap.add_argument("--override", action="append", default=[], metavar="k=v", help="hydra override (appended last)")
    ap.add_argument("--dry-run", action="store_true", help="print the environment and command, do not launch")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    if a.model:
        cfg["model_path"] = a.model
    if a.tasks:
        cfg["tasks"] = a.tasks
    set_overrides = split_sets(a.set, cfg)
    env, cmd, results = build(cfg)
    cmd += set_overrides + list(a.override)
    print_plan(env, cmd)
    if a.dry_run:
        return 0
    rc = stream_subprocess(cmd, env, line_hook=RunningPassAt1())
    summarize(results)
    return rc
