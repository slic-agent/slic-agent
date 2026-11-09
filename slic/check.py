"""`slic check` — the standard pre-run check of an installation, and the post-run check of a run's rollout dumps.

  slic check                                        Python packages, GPUs, the simulator binary and one real simulation
  slic check --config configs/train/slic_random.yaml   ... plus the config's files and the command it resolves to (dry run)
  slic check --dumps outputs/eval/<run>/rollouts [--expect-episodes N] [--partial-ok]
                                                    structure / schema / content of evaluation dump directories

Every item prints PASS, WARN or FAIL with the reason; the exit status is 1 if anything failed. GPUs are a WARN, not a
FAIL: `slic solve` with an API model and `slic serve` run without one, `slic train` and `slic eval` do not.
The simulation check resets the environment on the first task of a task parquet (default data/standard/test_iid.parquet,
`--tasks` for another) and requires finite gain and power from the simulator, which exercises the netlist writer, the
technology model card (`--tech-node`, default 65nm) and the result parser end to end.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import math
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from ._common import REPO_ROOT, ConfigError, load_yaml, print_plan, resolve_path

PACKAGES = ("torch", "vllm", "ray", "transformers", "verl", "fastapi")   # vllm / fastapi may be absent on a CPU-only install


class Report:
    def __init__(self) -> None:
        self.verdicts: List[str] = []

    def add(self, verdict: str, item: str, detail: str = "") -> None:
        self.verdicts.append(verdict)
        print(f"[{verdict}] {item}" + (f": {detail}" if detail else ""), flush=True)

    def summary(self) -> int:
        v = self.verdicts
        print(f"\n{v.count('PASS')} PASS / {v.count('WARN')} WARN / {v.count('FAIL')} FAIL")
        return 1 if "FAIL" in v else 0


# ----------------------------------------------------------------------------- installation
def check_python(rep: Report) -> None:
    if sys.version_info < (3, 10):
        rep.add("FAIL", "python", f"{sys.version.split()[0]} (3.10 or newer required)")
    else:
        rep.add("PASS", "python", f"{sys.version.split()[0]} at {sys.executable}")


def check_packages(rep: Report) -> None:
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception as e:   # ImportError or a broken install
            level = "WARN" if name in ("vllm", "fastapi") else "FAIL"
            rep.add(level, name, f"not importable ({type(e).__name__}: {e})")
            continue
        rep.add("PASS", name, str(getattr(mod, "__version__", "")) or "imported")


def check_gpus(rep: Report) -> None:
    try:
        import torch
    except Exception:
        rep.add("WARN", "gpu", "torch not importable, skipped")
        return
    if not torch.cuda.is_available():
        rep.add("WARN", "gpu", "no CUDA device (slic train / slic eval need one)")
        return
    names = sorted({torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())})
    rep.add("PASS", "gpu", f"{torch.cuda.device_count()} x {', '.join(names)}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")


def check_simulator(rep: Report) -> bool:
    backend = os.environ.get("CIRCUIT_SIM_BACKEND", "ngspice").strip().lower()
    if backend == "ngspice":
        binary = os.environ.get("NGSPICE_BIN", "ngspice")
        path = shutil.which(binary)
        if path is None:
            rep.add("FAIL", "simulator", f"ngspice binary not found: {binary!r} (install ngspice or set NGSPICE_BIN)")
            return False
        try:
            out = subprocess.run([path, "-v"], capture_output=True, text=True, timeout=30).stdout
            version = next((ln.strip(" *") for ln in out.splitlines() if "ngspice" in ln.lower()), "").strip()
        except Exception as e:
            rep.add("FAIL", "simulator", f"{path} -v failed: {e}")
            return False
        rep.add("PASS", "simulator", f"{version or 'ngspice'} ({path})")
        return True
    if backend == "spectre":
        path = shutil.which("spectre")
        if path is None:
            rep.add("FAIL", "simulator", "CIRCUIT_SIM_BACKEND=spectre but no spectre binary on PATH")
            return False
        rep.add("PASS", "simulator", f"spectre ({path})")
        return True
    rep.add("FAIL", "simulator", f"unknown CIRCUIT_SIM_BACKEND={backend!r} (ngspice or spectre)")
    return False


def check_simulation(rep: Report, tasks: Optional[str], tech_node: str) -> None:
    """Reset the environment on one released task: one real simulation through the whole stack."""
    path = resolve_path(tasks) if tasks else str(REPO_ROOT / "data" / "standard" / "test_iid.parquet")
    if not os.path.isfile(path):
        rep.add("WARN", "simulation", f"no task parquet at {path} (slic data --family standard, or --tasks <parquet>); skipped")
        return
    try:
        from .solve import REWARD_CFG, _goal_specs, _load_task_row
        from agent_system.environments.env_package.circuitop.circuitop.env import CircuitOpEnv
        from agent_system.environments.env_package.circuitop.reward_config import load_reward_config
    except Exception as e:
        rep.add("FAIL", "simulation", f"environment package not importable ({type(e).__name__}: {e})")
        return
    os.environ["CIRCUITOP_TECH_NODE"] = tech_node
    row = _load_task_row(path, 0, None)
    reward_cfg_path = str(REPO_ROOT / "agent_system" / "environments" / "env_package" / "circuitop" / REWARD_CFG) + ":sparse_constraint_done_4spec_hint"
    workdir = tempfile.mkdtemp(prefix="slic_check_")
    try:
        reward_config = load_reward_config(reward_cfg_path)
        env = CircuitOpEnv(workdir=workdir, max_steps=1, circuit_name=row["topology"], goal=_goal_specs(row["target"], None),
                           reward_config=reward_config)
        xml = (row["task_data"].get("structure") or {}).get("xml") or ""
        _, info = env.reset(initial_obs_xml=xml, task_data=row["task_data"], representation="xml")
    except Exception as e:
        rep.add("FAIL", "simulation", f"{row['topology']} ({row.get('task_id')}): {type(e).__name__}: {e}")
        return
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    gain = info.get("initial_gain_dB", info.get("gain_dB"))
    power = info.get("initial_power_uW", info.get("power_uW"))
    finite = lambda v: isinstance(v, (int, float)) and math.isfinite(float(v))   # noqa: E731
    if not (finite(gain) and finite(power)):
        rep.add("FAIL", "simulation", f"{row['topology']} ({row.get('task_id')}): non-finite initial metrics gain={gain} power={power}")
        return
    rep.add("PASS", "simulation", f"{row['topology']} ({row.get('task_id')}) at {tech_node}: gain {float(gain):.2f} dB, power {float(power):.2f} uW")


def check_config(rep: Report, config: str) -> None:
    """The config's files exist and its command resolves (the same dry run as `slic train/eval --dry-run`)."""
    cfg = load_yaml(config)
    try:
        if cfg.get("kind") == "sft":
            from .train import build_sft
            env, cmd = build_sft(cfg, dry_run=True)
        elif cfg.get("kind") == "rl":
            from .train import build_rl
            env, cmd = build_rl(cfg, dry_run=True)
        elif "model_path" in cfg and "tasks" in cfg:
            from .evaluate import build
            env, cmd, _ = build(cfg)
        else:
            rep.add("FAIL", "config", f"{config}: not a train (kind: rl | sft) or eval (model_path + tasks) config")
            return
    except ConfigError as e:
        rep.add("FAIL", "config", f"{config}: {str(e).replace('[slic] config error: ', '')}")
        return
    print_plan(env, cmd)
    rep.add("PASS", "config", f"{config} resolves")


# ----------------------------------------------------------------------------- rollout dumps
REQUIRED_KEYS = [
    "input", "output", "episode_length", "performance_success", "is_action_valid",
    "goal_gain_min", "goal_power_max", "gain_dB", "power_uW",
]
ACTION_RE = re.compile(r"<action>.*?</action>", re.S)


def validate_dump_dir(d: Path, expect: Optional[int], partial_ok: bool) -> Tuple[str, List[str]]:
    """Content-level validation of one evaluation dump directory (batch_*.jsonl step rows):
    structure (files parse, episodes reconstruct), schema (REQUIRED_KEYS), format (<action> tag rate, think tags, response
    lengths), behaviour (valid-action ratio, padding, episode lengths), content (goal fields, simulator metrics on valid
    steps, performance-success rate) and volume (episode count against --expect-episodes)."""
    issues: List[str] = []
    files = sorted(d.glob("batch_*.jsonl"))
    if not files:
        return "FAIL", ["no batch_*.jsonl files"]
    records, bad_json = [], 0
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad_json += 1
    if bad_json:
        issues.append(f"{bad_json} unparseable JSON lines")
    if not records:
        return "FAIL", issues + ["no records"]

    missing = [k for k in REQUIRED_KEYS if k not in records[0]]
    if missing:
        issues.append(f"missing keys: {missing}")

    eps, pos = [], 0
    while pos < len(records):
        ln = max(1, int(records[pos].get("episode_length", 1)))
        ep = records[pos:pos + ln]
        if ep and not ep[0].get("is_padding_sample", 0):
            eps.append(ep)
        pos += ln
    n_pad = sum(1 for r in records if r.get("is_padding_sample", 0))

    steps = [r for r in records if not r.get("is_padding_sample", 0)]
    if not steps or not eps:
        return "FAIL", issues + ["only padding rows"]
    action_rate = sum(1 for r in steps if ACTION_RE.search(r.get("output", ""))) / len(steps)
    think_rate = sum(1 for r in steps if "<think>" in r.get("output", "")) / len(steps)
    valid_rate = sum(int(r.get("is_action_valid", 0)) for r in steps) / len(steps)
    resp_lens = [r.get("turn_response_length") or len(r.get("output", "")) for r in steps]
    goal_null = sum(1 for r in steps if r.get("goal_gain_min") is None) / len(steps)
    valid_steps = [r for r in steps if r.get("is_action_valid")]
    metr_null = (sum(1 for r in valid_steps if r.get("gain_dB") is None) / len(valid_steps)) if valid_steps else 1.0
    perf_rate = sum(1 for ep in eps if any(s.get("performance_success") for s in ep)) / len(eps)
    ep_lens = [len(ep) for ep in eps]

    # valid-action ratio: < 10 % indicates format / output collapse, 10-35 % is worth a look (design errors are normal
    # exploration)
    hard_issues: List[str] = []
    if action_rate < 0.9:
        hard_issues.append(f"action-tag rate {action_rate:.0%} < 90% (format collapse)")
    if valid_rate < 0.10:
        hard_issues.append(f"valid-action ratio {valid_rate:.0%} < 10% (collapse)")
    elif valid_rate < 0.35:
        issues.append(f"valid-action ratio {valid_rate:.0%} below healthy baseline (~46%)")
    if goal_null > 0:
        hard_issues.append(f"{goal_null:.0%} steps missing goal fields")
    if metr_null > 0.05:
        hard_issues.append(f"{metr_null:.0%} valid steps missing sim metrics")
    if expect is not None and len(eps) < expect:
        msg = f"episodes {len(eps)}/{expect}"
        if partial_ok:
            issues.append("(partial) " + msg)
        else:
            return "FAIL", issues + [msg]
    if perf_rate == 0 and len(eps) > 50:
        issues.append("performance-success rate 0 over >50 episodes (suspicious)")

    if missing:
        hard_issues.append("schema")
    issues = hard_issues + issues
    verdict = "FAIL" if hard_issues else ("WARN" if issues else "PASS")
    print(f"[{verdict}] {d.parent.name[:64]}: {len(eps)} eps, action {action_rate:.0%}, think {think_rate:.0%}, "
          f"valid {valid_rate:.0%}, perf-success {perf_rate:.1%}, resp_len p50 {int(st.median(resp_lens))}, "
          f"ep_len p50 {int(st.median(ep_lens))}, pad {n_pad}")
    for i in issues:
        print(f"        - {i}")
    return verdict, issues


def check_dumps(rep: Report, patterns: List[str], expect: Optional[int], partial_ok: bool) -> None:
    dirs = [Path(p) for pat in patterns for p in sorted(glob.glob(pat))]
    if not dirs:
        raise ConfigError(f"--dumps: nothing matches {' '.join(patterns)}")
    for d in dirs:
        rep.verdicts.append(validate_dump_dir(d, expect, partial_ok)[0])


# ----------------------------------------------------------------------------- entry point
def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="also check this train / eval config (files and resolved command)")
    ap.add_argument("--tasks", help="task parquet for the simulation check (default data/standard/test_iid.parquet)")
    ap.add_argument("--tech-node", default="65nm", help="technology node of the simulation check")
    ap.add_argument("--no-simulation", action="store_true", help="skip the simulation check")
    ap.add_argument("--dumps", nargs="+", metavar="DIR", help="evaluation rollout dump directories (globs); checks only these")
    ap.add_argument("--expect-episodes", type=int, default=None, help="--dumps: episodes each directory must hold")
    ap.add_argument("--partial-ok", action="store_true", help="--dumps: an episode shortfall is a warning (run still going)")
    a = ap.parse_args(argv)
    rep = Report()
    if a.dumps:
        check_dumps(rep, a.dumps, a.expect_episodes, a.partial_ok)
        return rep.summary()
    check_python(rep)
    check_packages(rep)
    check_gpus(rep)
    if check_simulator(rep) and not a.no_simulation:
        check_simulation(rep, a.tasks, a.tech_node)
    if a.config:
        check_config(rep, a.config)
    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(main())
