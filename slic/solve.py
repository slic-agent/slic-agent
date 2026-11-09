"""`slic solve --config task.yaml` — run one optimization episode and print every turn as it happens.

The episode goes through the same CircuitOp environment, simulator backend and reward
configuration as training/eval; the policy is queried through an OpenAI-compatible server
(e.g. `vllm serve <checkpoint> --limit-mm-per-prompt '{"image":0,"video":0}'` for a Qwen3.5 checkpoint), so this command itself needs no GPU.

Config keys:
  task:
    tasks: data/standard/test_iid.parquet     # a task parquet (your own circuits: build one with `slic tasks`) ...
    index: 0                                  # ... and the row to solve (or task_id: <id>)
    # goal: {gain_min: 40, power_max: 300, bandwidth_min: 1.0, phase_margin_min_deg: 60}   # override the row's specs
  model:
    provider: local                           # local (OpenAI-compatible) | openai | gemini | claude | deepseek | openrouter
    base_url: http://127.0.0.1:8000/v1
    name: slic-random                         # the --served-model-name of the vllm server
    # temperature / top_p / max_tokens / seed: sampling (default: the evaluation protocol for local, the agent's for APIs;
    # without a seed every run samples afresh, with one a local vLLM server repeats the episode)
  max_turns: 8
  reward_preset: sparse_constraint_done_4spec_hint
  tech_node: 65nm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from ._common import ConfigError, check_model_endpoint, load_yaml, resolve_path

REWARD_CFG = "configs/reward_config_example.yaml"   # relative to the circuitop env package


def _load_task_row(path: str, index: Optional[int], task_id: Optional[str]) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_parquet(path)
    if task_id is not None:
        rows = df[df["extra_info"].apply(lambda e: str(e.get("task_id")) == str(task_id))]
        if rows.empty:
            raise ConfigError(f"task_id {task_id!r} not found in {path}")
        row = rows.iloc[0]
    else:
        row = df.iloc[int(index or 0)]
    ek = dict(row["extra_info"]["env_kwargs"])
    return {
        "task_data": dict(ek["task_data"]),
        "topology": ek["topology"],
        "target": dict(ek.get("target") or {}),
        "task_id": row["extra_info"].get("task_id"),
    }


def _goal_specs(target: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("gain_min", "power_max", "bandwidth_min", "phase_margin_min_deg")
    specs = {k: target[k] for k in keys if target.get(k) is not None}
    for k, v in (override or {}).items():
        if v is not None:
            specs[k] = v
    return specs


def _v(x: Optional[float], spec: str) -> str:
    """a metric, or n/a where the environment reports it as not measurable (None)"""
    return format(x, spec) if x is not None else "n/a".rjust(int(spec.split(".")[0]) if spec[0].isdigit() else 0)


def _bw(x: Optional[float]) -> str:
    """bandwidth (MHz) in a readable unit: 12.34 MHz / 552.3 kHz / 830 Hz; n/a when not measurable"""
    if x is None:
        return "n/a"
    return f"{x:.2f} MHz" if x >= 1 else (f"{x * 1e3:.1f} kHz" if x >= 1e-3 else f"{x * 1e6:.0f} Hz")


def _fmt(step: Dict[str, Any]) -> str:
    flag = "ok " if step.get("valid") else "INV"
    mark = " *best" if step.get("is_best") else ""
    goal = " GOAL-MET" if step.get("goal_met") else ""
    # score = the per-turn score of the post-action state (perf_score_post: what training rewards and best_turn ranks);
    # the environment's episode reward is 0 until the last turn, so it is not printed per turn
    return (f"turn {step['step'] + 1:2d} [{flag}] gain={_v(step['gain_dB'], '7.2f')} dB  power={_v(step['power_uW'], '9.2f')} uW  "
            f"bw={_bw(step['bandwidth_MHz']):>11s}  pm={_v(step['phase_margin_deg'], '6.1f')} deg  "
            f"score={step['perf_score_post']:+.3f}{goal}{mark}")


def run(cfg: Dict[str, Any], jsonl: Optional[str], quiet: bool) -> int:
    task_cfg = cfg.get("task") or {}
    model_cfg = cfg.get("model") or {}
    tech_node = str(cfg.get("tech_node", "65nm"))

    if "tasks" not in task_cfg:
        # a task is a row of a task parquet: it carries the netlist the simulator needs (an XML structure alone does not)
        raise ConfigError("task needs tasks (+ index or task_id); for your own circuit, build a task parquet with `slic tasks`")
    row = _load_task_row(resolve_path(task_cfg["tasks"]), task_cfg.get("index"), task_cfg.get("task_id"))
    task_data, topology, target = row["task_data"], row["topology"], row["target"]
    label = row.get("task_id") or f"{topology}#{task_cfg.get('index', 0)}"

    provider = str(model_cfg.get("provider", "local")).lower()
    preset = cfg.get("reward_preset", "sparse_constraint_done_4spec_hint")

    from web_ui.streaming_inference import run_streaming_inference

    # the evaluation pipeline with the protocol of scripts/eval/run_eval.sh (the model sees exactly the `slic eval` prompts);
    # max_turns / reward_preset / tech_node / model.temperature override it
    events = run_streaming_inference(
        task={"topology": topology, "target": target, "task_data": task_data},
        provider=provider,
        model_name=model_cfg.get("name"),
        temperature=model_cfg.get("temperature"),
        top_p=model_cfg.get("top_p"),
        max_tokens=model_cfg.get("max_tokens"),
        seed=model_cfg.get("seed"),
        max_steps=int(cfg.get("max_turns", 8)),
        reward_preset=preset,
        tech_node=tech_node,
        local_model_base_url=model_cfg.get("base_url") if provider == "local" else None,
        local_model_name=model_cfg.get("name") if provider == "local" else None,
        goal_override=task_cfg.get("goal"),
    )
    out = open(jsonl, "w", encoding="utf-8") if jsonl else None
    rc = 0
    print(f"[slic] task={label} topology={topology} tech={tech_node} backend={os.environ.get('CIRCUIT_SIM_BACKEND', 'ngspice')} preset={preset}")
    streaming = None   # the turn whose reply is being printed live
    for raw in events:
        ev = json.loads(raw)
        kind = ev.get("event")
        if out and kind != "token":   # the reply is in the step event; the jsonl keeps one record per turn
            out.write(raw + "\n")
            out.flush()
        if kind == "token":
            if quiet:
                continue
            if streaming != ev["step"]:
                streaming = ev["step"]
                print(f"\n----- turn {ev['step'] + 1}: model reply (streaming) -----", flush=True)
            sys.stdout.write(ev["text"]); sys.stdout.flush()
            continue
        if streaming is not None:
            print("\n" + "-" * 44, flush=True); streaming = None
        if kind == "init":
            print(f"[slic] goal: {ev['goal_display']}")
            print(f"[slic] sampling: {ev.get('sampling')}")
            print(f"[slic] start: gain={_v(ev['initial_gain_dB'], '.2f')} dB power={_v(ev['initial_power_uW'], '.2f')} uW "
                  f"bw={_bw(ev['initial_bandwidth_MHz'])} pm={_v(ev['initial_phase_margin_deg'], '.1f')} deg")
        elif kind == "step":
            if "error" in ev:
                print(f"turn {ev['step'] + 1:2d} ERROR {ev['error']}")
                rc = 2
                continue
            print(_fmt(ev), flush=True)
            if not quiet and ev.get("action"):
                print(f"         action: {ev['action']}")
            if not quiet and ev.get("env_error"):
                print(f"         env: {ev['env_error'][:200]}")
        elif kind == "done":
            print(f"[slic] done: turns={ev['total_steps']} success={ev['success']} best_turn={ev['best_step_idx'] + 1 if ev['best_step_idx'] >= 0 else '-'}")
            if cfg.get("print_netlist", False):
                print(ev.get("best_netlist", ""))
        elif kind == "error":
            print(f"[slic] error: {ev.get('message')}")
            rc = 1
    if out:
        out.close()
        print(f"[slic] events written to {jsonl}")
    return rc


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic solve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", type=int, help="override task.index")
    ap.add_argument("--jsonl", help="also write the events here, one JSON per line (init / step / done; each step event holds the turn's full reply, so the streamed token events are not repeated)")
    ap.add_argument("--quiet", action="store_true", help="print only the per-turn summary lines (no live model reply, no actions)")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    if a.index is not None:
        cfg.setdefault("task", {})["index"] = a.index
    sys.path.insert(0, str(__import__("slic._common", fromlist=["REPO_ROOT"]).REPO_ROOT))
    m = cfg.get("model") or {}
    if str(m.get("provider", "local")).lower() == "local":   # the endpoint must serve this model before the episode starts
        check_model_endpoint(m.get("base_url") or "", m.get("name") or "")
    return run(cfg, a.jsonl, a.quiet)
