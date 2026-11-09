"""`slic train --config configs/train/<arm>.yaml` — launch an RL (SLIC / fixed-base baseline / MT-GRPO) or SFT run.

RL (kind: rl) runs scripts/train/run_slic.sh, whose command line holds the paper's settings of the main method; a config lists
only what differs from it. Config keys and the hydra overrides they become:
  experiment_name      trainer.experiment_name (outputs/{checkpoints,train_dumps,sim_workdir}/<name>)
  model_path           actor_rollout_ref.model.path — the SFT checkpoint (HF format) to start from (required)
  dataset_dir          data.train_files / data.val_files = its train.parquet / validation.parquet (default data/standard);
                       train_file / val_file set the two files directly
  num_gpus             trainer.n_gpus_per_node (default: the visible GPUs)
  output_root          directory holding checkpoints/, train_dumps/ and sim_workdir/ (default outputs/)
  wandb (bool)         trainer.logger=['console','wandb']        project_name   trainer.project_name
  reward_preset        env.circuitop.reward_config=<reward_config_example.yaml>:<preset>
  overrides: [k=v]     hydra overrides appended after those (later ones win); env: {VAR: value} runtime environment variables
SFT (kind: sft) runs scripts/train/sft/run_sft.sh the same way: experiment_name -> trainer.experiment_name, base_model ->
  model.partial_pretrain, train_file / val_file -> data.train_files / data.val_files, num_gpus -> NUM_GPUS (torchrun processes),
  save_path -> trainer.default_local_dir, chat_template -> +model.chat_template_path, wandb / project_name as above.

Command line: `--set KEY=VALUE` sets a config key above (dataset_dir=data/hard, num_gpus=4) and is a hydra override otherwise
(trainer.total_training_steps=10); `--override k=v` is always a hydra override. Command-line overrides come after the config's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from typing import Dict, List

import yaml

from ._common import REPO_ROOT, ConfigError, load_yaml, merge_env, print_plan, resolve_path, stream_subprocess

REWARD_CONFIG = "configs/reward_config_example.yaml"   # resolved by RewardConfig.from_yaml against the circuitop env package
RL_KEYS = {"experiment_name", "model_path", "dataset_dir", "train_file", "val_file", "num_gpus", "output_root", "wandb", "project_name",
           "reward_preset"}
SFT_KEYS = {"experiment_name", "base_model", "train_file", "val_file", "num_gpus", "save_path", "chat_template", "wandb", "project_name"}
RETIRED_PREFIXES = ("SLIC_", "SFT_", "EVAL_")   # SLIC_* / SFT_* / EVAL_* variables are rejected: the launchers take no settings from the environment


def check_env(env: Dict[str, str], where: str) -> None:
    stale = sorted(k for k in env if k.startswith(RETIRED_PREFIXES))
    if stale:
        raise ConfigError(f"{where}: {', '.join(stale)} — the launchers read no settings from the environment; use the config keys "
                          "(model_path, num_gpus, ...) or hydra overrides (e.g. trainer.total_training_steps=10)")


def gpu_count(dry_run: bool) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([d for d in visible.split(",") if d.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        out = ""
    n = len([line for line in out.splitlines() if line.startswith("GPU")])
    if n == 0:
        if dry_run:
            return 1
        raise ConfigError("no GPU visible; set num_gpus in the config or --set num_gpus=N")
    return n


def _check_file(path: str, hint: str) -> str:
    if not os.path.isfile(path):
        raise ConfigError(f"missing parquet: {path} ({hint})")
    return path


def build_rl(cfg: Dict, dry_run: bool = False) -> tuple[Dict[str, str], List[str]]:
    if not cfg.get("model_path"):
        raise ConfigError("model_path is not set: the paper's SFT initialisations are not released, so put the path of your own "
                          "SFT checkpoint (HF format) in the config's model_path or pass --set model_path=<dir> "
                          "(see scripts/train/sft/README.md to build one)")
    ov: List[str] = []
    if cfg.get("experiment_name"):
        ov.append(f"trainer.experiment_name={cfg['experiment_name']}")
    ov.append(f"actor_rollout_ref.model.path={resolve_path(cfg['model_path'])}")
    dataset_dir = resolve_path(cfg.get("dataset_dir") or "data/standard")
    train = resolve_path(cfg["train_file"]) if cfg.get("train_file") else f"{dataset_dir}/train.parquet"
    val = resolve_path(cfg["val_file"]) if cfg.get("val_file") else f"{dataset_dir}/validation.parquet"
    hint = "run: slic data --family standard"
    ov += [f"data.train_files={_check_file(train, hint)}", f"data.val_files=[{_check_file(val, hint)}]"]
    ov.append(f"trainer.n_gpus_per_node={cfg['num_gpus'] if cfg.get('num_gpus') else gpu_count(dry_run)}")
    if cfg.get("output_root"):
        root = resolve_path(cfg["output_root"])
        ov += [f"trainer.default_local_dir={root}/checkpoints/${{trainer.experiment_name}}",
               f"trainer.rollout_data_dir={root}/train_dumps/${{trainer.experiment_name}}",
               f"trainer.validation_data_dir={root}/train_dumps/${{trainer.experiment_name}}/val",
               f"env.circuitop.workdir={root}/sim_workdir/${{trainer.experiment_name}}"]
    if cfg.get("wandb"):
        ov.append("trainer.logger=['console','wandb']")
    if cfg.get("project_name"):
        ov.append(f"trainer.project_name={cfg['project_name']}")
    if cfg.get("reward_preset"):
        ov.append(f"env.circuitop.reward_config={REWARD_CONFIG}:{cfg['reward_preset']}")
    env = merge_env(cfg.get("env"))
    check_env(env, "config env:")
    cmd = ["bash", str(REPO_ROOT / "scripts" / "train" / "run_slic.sh")] + ov + [str(o) for o in cfg.get("overrides") or []]
    return env, cmd


def build_sft(cfg: Dict, dry_run: bool = False) -> tuple[Dict[str, str], List[str]]:
    ov: List[str] = []
    if cfg.get("experiment_name"):
        ov.append(f"trainer.experiment_name={cfg['experiment_name']}")
    if cfg.get("base_model"):
        ov.append(f"model.partial_pretrain={resolve_path(cfg['base_model'])}")
    hint = "a parquet with a messages column, see scripts/train/sft/README.md"
    for key, hydra_key in (("train_file", "data.train_files"), ("val_file", "data.val_files")):
        if cfg.get(key):
            ov.append(f"{hydra_key}={_check_file(resolve_path(cfg[key]), hint)}")
    if cfg.get("save_path"):
        ov.append(f"trainer.default_local_dir={resolve_path(cfg['save_path'])}")
    if cfg.get("chat_template"):
        ov.append(f"+model.chat_template_path={resolve_path(cfg['chat_template'])}")
    if cfg.get("wandb"):
        ov.append("trainer.logger=['console','wandb']")
    if cfg.get("project_name"):
        ov.append(f"trainer.project_name={cfg['project_name']}")
    env = merge_env(cfg.get("env"))
    check_env(env, "config env:")
    env["NUM_GPUS"] = str(cfg["num_gpus"] if cfg.get("num_gpus") else gpu_count(dry_run))
    cmd = ["bash", str(REPO_ROOT / "scripts" / "train" / "sft" / "run_sft.sh")] + ov + [str(o) for o in cfg.get("overrides") or []]
    return env, cmd


def split_sets(sets: List[str], keys: set, cfg: Dict) -> List[str]:
    """--set KEY=VALUE: a config key updates cfg (typed like the YAML: num_gpus=4, wandb=false); anything else is a hydra override."""
    overrides: List[str] = []
    for item in sets:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if k in keys:
            cfg[k] = yaml.safe_load(v) if v else None
        elif k.startswith(RETIRED_PREFIXES):
            check_env({k: v}, "--set")
        else:
            overrides.append(item)
    return overrides


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic train", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a config key (model_path=...) or else a hydra override")
    ap.add_argument("--override", action="append", default=[], metavar="k=v", help="hydra override (appended last)")
    ap.add_argument("--dry-run", action="store_true", help="print the environment and command, do not launch")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    kind = str(cfg.get("kind", "rl")).lower()
    if kind not in ("rl", "sft"):
        raise ConfigError(f"kind must be rl or sft, got {kind!r}")
    set_overrides = split_sets(a.set, RL_KEYS if kind == "rl" else SFT_KEYS, cfg)
    env, cmd = (build_rl if kind == "rl" else build_sft)(cfg, a.dry_run)
    cmd += set_overrides + list(a.override)
    print_plan(env, cmd)
    if a.dry_run:
        return 0
    return stream_subprocess(cmd, env)
