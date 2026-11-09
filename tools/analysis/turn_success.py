#!/usr/bin/env python3
"""Analyze turn-by-turn spec-combo success from eval rollout JSONL dumps.

This script reads `batch_*.jsonl` files produced by
`slic eval` (`scripts/eval/run_eval.sh`), reconstructs episodes from
the per-step records using `episode_length`, and computes for each spec combo:

- new success rate at exact turn position t
- state success rate among episodes that reached turn t
- cumulative success rate by turn t
- fraction of episodes that reached turn t

The analysis is post-hoc and uses the saved raw metrics/goals in each step
record, so it can report multiple spec combos from a single rollout run.
Invalid actions are always counted as not successful for their own turn, even
if the rollout record carries inherited metrics from the previous state.
"""

import argparse
import csv
import glob
import re
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional



# ── pass@k helpers ────────────────────────────────────────────────────
def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


# Per-spec verdicts for the partial combos (gain_pm, gain_power, gain_only); all_4 uses the environment's own
# performance_success. 5 % relative tolerance on every spec, as in the released reward presets.
_SPEC_TOL = 0.05


def compute_spec_pass(record: dict) -> dict:
    """Per-spec pass / fail of one dump row from its metrics and goal fields (5 % tolerances):
      gain >= goal - |goal| * 5 %;  power <= goal + |goal| * 5 %;  bw >= goal * (1 - 5 %);  pm >= goal * (1 - 5 %).
    pm passes without a goal; without a measured pm (no unity-gain crossing) it passes iff gain > 0. bw passes without
    a goal."""
    gain = record.get("gain_dB")
    pm = record.get("phase_margin_deg")
    power = record.get("power_uW")
    bw = record.get("bandwidth_MHz")

    goal_gain = record.get("goal_gain_min")
    goal_pm = record.get("goal_phase_margin_min_deg")
    goal_power = record.get("goal_power_max")
    goal_bw = record.get("goal_bandwidth_min")

    result = {}
    result["gain"] = (
        gain is not None and goal_gain is not None
        and float(gain) >= float(goal_gain) - abs(float(goal_gain)) * _SPEC_TOL
    )
    if goal_pm is None or float(goal_pm) <= 0:
        result["pm"] = True
    elif pm is None:
        result["pm"] = gain is not None and float(gain) > 0.0
    else:
        result["pm"] = float(pm) >= float(goal_pm) * (1.0 - _SPEC_TOL)
    result["power"] = (
        power is not None and goal_power is not None
        and float(power) <= float(goal_power) + abs(float(goal_power)) * _SPEC_TOL
    )
    if goal_bw is None or float(goal_bw) <= 0:
        result["bw"] = True
    else:
        result["bw"] = bw is not None and float(bw) >= float(goal_bw) * (1.0 - _SPEC_TOL)
    return result


def _pass_at_k(n: int, c: int, k: int) -> Optional[float]:
    """Unbiased pass@k estimator: 1 - C(n-c,k)/C(n,k). Returns None if k>n."""
    if k > n:
        return None
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - math.exp(_log_comb(n - c, k) - _log_comb(n, k))


def _input_hash(episode: List[dict]) -> str:
    text = episode[0].get("input", "") if episode else ""
    if isinstance(text, list):
        text = "".join(str(m.get("content", "")) for m in text if isinstance(m, dict))
    return hashlib.md5(text.encode()).hexdigest()[:16] if text else ""


COMBOS = {
    "all_4": ["gain", "power", "bw", "pm"],
    "gain_power_pm": ["gain", "power", "pm"],
    "gain_power_bw": ["gain", "power", "bw"],
    "gain_power": ["gain", "power"],
    "gain_bw": ["gain", "bw"],
    "gain_only": ["gain"],
}


COMBO_ORDER = list(COMBOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, help="Directory containing batch_*.jsonl rollout files")
    parser.add_argument("--output-json", required=True, help="Path to write JSON summary")
    parser.add_argument("--output-csv", required=True, help="Path to write CSV turn table")
    parser.add_argument("--output-markdown", required=True, help="Path to write Markdown summary")
    parser.add_argument("--case-label", default="", help="Checkpoint / experiment case label")
    parser.add_argument("--split", default="", help="Dataset split label")
    parser.add_argument("--eval-reward-label", default="", help="Eval reward label used for rollout generation")
    parser.add_argument("--history-length", type=int, default=None, help="History length used during eval")
    parser.add_argument("--max-steps", type=int, default=None, help="Configured max_steps during eval")
    parser.add_argument("--data-path", default="", help="Optional split parquet used to recover complexity metadata")
    parser.add_argument("--dataset-dir", default="", help="Optional dataset dir; uses <split>.parquet when --data-path is unset")
    parser.add_argument("--pass-k-values", default="1,5", help="Comma-separated k values for pass@k (default: 1,5)")
    return parser.parse_args()


def load_batch_records(rollout_dir: Path) -> List[dict]:
    records: List[dict] = []
    # Numeric sort: lexical sorting would put batch_10 before batch_2, scrambling
    # episode order vs generation order and the positional match to the parquet
    batch_files = sorted(
        glob.glob(str(rollout_dir / "batch_*.jsonl")),
        key=lambda p: int(re.search(r"batch_(\d+)", Path(p).name).group(1)),
    )
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.jsonl files found under {rollout_dir}")

    for file_path in batch_files:
        with open(file_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def split_into_episodes(records: List[dict]) -> List[List[dict]]:
    episodes: List[List[dict]] = []
    pos = 0
    while pos < len(records):
        episode_length = int(records[pos].get("episode_length", 1))
        if episode_length <= 0:
            episode_length = 1
        end = min(pos + episode_length, len(records))
        episode = records[pos:end]
        if episode and not episode[0].get("is_padding_sample", 0):
            episodes.append(episode)
        pos = end
    return episodes


def is_valid_action_step(step: dict) -> bool:
    value = step.get("is_action_valid", 1)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "invalid"}
    return value not in (0, False)


def _safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def infer_complexity(topology: Optional[str], explicit_complexity: Optional[str] = None) -> str:
    topo = str(topology or "").strip().lower().replace("-", "_").replace(" ", "_")
    if topo.startswith("single_ended") or topo.startswith("fully_differential"):
        return "complex"
    if topo:
        return "simple"

    explicit = str(explicit_complexity or "").strip().lower()
    if explicit in {"simple", "complex"}:
        return explicit
    return "unknown"


def _nested_get(obj, keys: List[str]):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def extract_step_complexity(step: dict) -> str:
    explicit = (
        step.get("topology_complexity")
        or step.get("complexity")
        or _nested_get(step, ["metadata", "complexity"])
    )
    topology = step.get("topology") or step.get("circuit_name") or _nested_get(step, ["metadata", "topology_name"])
    return infer_complexity(topology, explicit)


def _prompt_to_text(prompt) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    parts = []
    try:
        iterator = iter(prompt)
    except TypeError:
        return str(prompt)
    for item in iterator:
        if isinstance(item, dict):
            parts.append(str(item.get("content", "")))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _prompt_signatures(prompt_text: str) -> List[str]:
    signatures = []
    for marker in ("_IDMAP>", "<IDMAP", "type=pin:"):
        idx = prompt_text.find(marker)
        if idx >= 0:
            sig = prompt_text[idx : idx + 1200].strip()
            if len(sig) >= 200:
                signatures.append(sig)
            break
    first_chunk = prompt_text[:1200].strip()
    if len(first_chunk) >= 200:
        signatures.append(first_chunk)
    return list(dict.fromkeys(signatures))


def extract_parquet_complexity_metadata(data_path: Optional[Path]) -> dict:
    if not data_path:
        return {"ordered_complexities": [], "prompt_signatures": []}
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {data_path}")

    import pandas as pd

    df = pd.read_parquet(data_path)
    complexities = []
    prompt_signatures = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            complexities.append("unknown")
            continue
        explicit = (
            extra_info.get("complexity")
            or _nested_get(extra_info, ["metadata", "complexity"])
            or _nested_get(extra_info, ["env_kwargs", "topology_complexity"])
            or _nested_get(extra_info, ["env_kwargs", "task_data", "metadata", "complexity"])
        )
        topology = (
            extra_info.get("topology")
            or _nested_get(extra_info, ["env_kwargs", "topology"])
            or _nested_get(extra_info, ["metadata", "topology_name"])
        )
        complexity = infer_complexity(topology, explicit)
        complexities.append(complexity)
        prompt_text = _prompt_to_text(row.get("prompt"))
        for signature in _prompt_signatures(prompt_text):
            prompt_signatures.append(
                {
                    "signature": signature,
                    "complexity": complexity,
                    "topology": str(topology or ""),
                }
            )
    return {"ordered_complexities": complexities, "prompt_signatures": prompt_signatures}


def extract_parquet_complexities(data_path: Optional[Path]) -> List[str]:
    return extract_parquet_complexity_metadata(data_path)["ordered_complexities"]


def combo_passes_for_episode(episode_steps: List[dict]) -> Dict[str, List[bool]]:
    combo_passes: Dict[str, List[bool]] = {name: [] for name in COMBOS}
    for step in episode_steps:
        if not is_valid_action_step(step):
            spec_pass = {}
            all_4_pass = False
        else:
            spec_pass = compute_spec_pass(step)
            # Use the env-computed performance_success for all_4 when available;
            # it applies the exact tolerances used during evaluation.
            perf = step.get("performance_success")
            all_4_pass = bool(perf) if perf is not None else all(spec_pass.get(s, False) for s in COMBOS["all_4"])
        for combo_name, combo_specs in COMBOS.items():
            if combo_name == "all_4":
                combo_passes[combo_name].append(all_4_pass)
            else:
                combo_passes[combo_name].append(all(spec_pass.get(spec, False) for spec in combo_specs))
    return combo_passes


def first_success_turn(pass_list: List[bool]) -> Optional[int]:
    for idx, passed in enumerate(pass_list, start=1):
        if passed:
            return idx
    return None


def resolve_episode_complexity(
    episode: List[dict],
    episode_idx: int,
    total_episodes: int,
    dataset_metadata: Optional[dict],
) -> str:
    complexity = extract_step_complexity(episode[0]) if episode else "unknown"
    if complexity != "unknown":
        return complexity

    dataset_metadata = dataset_metadata or {}
    episode_input = str((episode[0] or {}).get("input") or "") if episode else ""
    if episode_input:
        for item in dataset_metadata.get("prompt_signatures", []):
            signature = item.get("signature")
            if signature and signature in episode_input:
                return item.get("complexity", "unknown")

    # Last resort: positional match when the episode count equals the parquet's. Prefer unknown over
    # fabricating simple/complex when the rollout order cannot be matched to parquet.
    ordered = dataset_metadata.get("ordered_complexities", [])
    if len(ordered) == total_episodes and episode_idx < len(ordered):
        return ordered[episode_idx]
    return "unknown"


def _build_complexity_turn_rows(
    episode_summaries: List[dict],
    pass_k_values: List[int],
    max_turn_observed: int,
) -> List[dict]:
    """Per-complexity breakdown of turn-level pass rates."""
    rows = []
    for complexity in ["simple", "complex", "unknown"]:
        group = [s for s in episode_summaries if s["complexity"] == complexity]
        if not group:
            continue
        group_total = len(group)
        max_turn_in_group = max(s["episode_length"] for s in group)

        cx_prompt_groups: List[List[dict]] = []
        if pass_k_values:
            _cx_groups: Dict[str, List[dict]] = defaultdict(list)
            for s in group:
                _cx_groups[s["input_hash"]].append(s)
            cx_prompt_groups = list(_cx_groups.values())

        for combo_name in COMBOS:
            for turn in range(1, max_turn_in_group + 1):
                episodes_reached = 0
                at_successes = 0
                by_successes = 0

                for summary in group:
                    if summary["episode_length"] >= turn:
                        episodes_reached += 1
                        if summary["combo_passes"][combo_name][turn - 1]:
                            at_successes += 1
                    fst = summary["first_success_turn"][combo_name]
                    if fst is not None and fst <= turn:
                        by_successes += 1

                pass_k_row: Dict[str, Optional[float]] = {}
                if pass_k_values and cx_prompt_groups:
                    for k_val in pass_k_values:
                        by_vals: List[float] = []
                        at_vals: List[float] = []
                        for grp in cx_prompt_groups:
                            n_by = len(grp)
                            c_by = sum(
                                1 for s in grp
                                if s["first_success_turn"][combo_name] is not None
                                and s["first_success_turn"][combo_name] <= turn
                            )
                            v_by = _pass_at_k(n_by, c_by, k_val)
                            if v_by is not None:
                                by_vals.append(v_by)
                            reached_grp = [s for s in grp if s["episode_length"] >= turn]
                            n_at = len(reached_grp)
                            c_at = sum(1 for s in reached_grp if s["combo_passes"][combo_name][turn - 1])
                            v_at = _pass_at_k(n_at, c_at, k_val)
                            if v_at is not None:
                                at_vals.append(v_at)
                        pass_k_row[f"pass_at_{k_val}_by_turn"] = sum(by_vals) / len(by_vals) if by_vals else None
                        pass_k_row[f"pass_at_{k_val}_at_turn"] = sum(at_vals) / len(at_vals) if at_vals else None

                rows.append(
                    {
                        "complexity": complexity,
                        "combo": combo_name,
                        "turn": turn,
                        "episodes_total": group_total,
                        "episodes_reached": episodes_reached,
                        "reached_rate": (episodes_reached / group_total) if group_total else 0.0,
                        "success_rate_at_turn": (at_successes / group_total) if group_total else 0.0,
                        "success_rate_by_turn": (by_successes / group_total) if group_total else 0.0,
                        **pass_k_row,
                    }
                )
    return rows


def aggregate_episode_stats(
    episodes: List[List[dict]],
    dataset_metadata: Optional[dict] = None,
    pass_k_values: Optional[List[int]] = None,
) -> dict:
    episode_summaries = []
    max_turn_observed = 0

    dataset_metadata = dataset_metadata or {}
    pass_k_values = pass_k_values or []

    for episode_idx, episode in enumerate(episodes):
        combo_passes = combo_passes_for_episode(episode)
        episode_length = len(episode)
        max_turn_observed = max(max_turn_observed, episode_length)
        complexity = resolve_episode_complexity(episode, episode_idx, len(episodes), dataset_metadata)
        first_success = {
            combo_name: first_success_turn(pass_list)
            for combo_name, pass_list in combo_passes.items()
        }
        _raw_input = episode[0].get("input", "") if episode else ""
        _prompt_chars = len(_prompt_to_text(_raw_input))
        episode_summaries.append(
            {
                "episode_length": episode_length,
                "combo_passes": combo_passes,
                "first_success_turn": first_success,
                "complexity": complexity,
                "input_hash": _input_hash(episode),
                "prompt_chars": _prompt_chars,
            }
        )

    # Build prompt groups for pass@k (group summaries by their step-0 input hash)
    prompt_groups: List[List[dict]] = []
    if pass_k_values:
        _groups: Dict[str, List[dict]] = defaultdict(list)
        for s in episode_summaries:
            _groups[s["input_hash"]].append(s)
        prompt_groups = list(_groups.values())

    total_episodes = len(episode_summaries)
    turn_rows = []
    combo_summary = {}
    complexity_summary = {}

    for combo_name in COMBOS:
        combo_rows = []
        first_success_hist = {}
        never_succeeded = 0

        for summary in episode_summaries:
            fst = summary["first_success_turn"][combo_name]
            if fst is None:
                never_succeeded += 1
            else:
                first_success_hist[fst] = first_success_hist.get(fst, 0) + 1

        for turn in range(1, max_turn_observed + 1):
            episodes_reached = 0
            at_successes = 0
            first_successes = 0
            by_successes = 0

            for summary in episode_summaries:
                episode_length = summary["episode_length"]
                if episode_length >= turn:
                    episodes_reached += 1
                    if summary["combo_passes"][combo_name][turn - 1]:
                        at_successes += 1

                fst = summary["first_success_turn"][combo_name]
                if fst == turn:
                    first_successes += 1
                if fst is not None and fst <= turn:
                    by_successes += 1

            pass_k_row: Dict[str, Optional[float]] = {}
            if pass_k_values and prompt_groups:
                for k_val in pass_k_values:
                    by_vals: List[float] = []
                    at_vals: List[float] = []
                    for group in prompt_groups:
                        n_by = len(group)
                        c_by = sum(
                            1 for s in group
                            if s["first_success_turn"][combo_name] is not None
                            and s["first_success_turn"][combo_name] <= turn
                        )
                        v_by = _pass_at_k(n_by, c_by, k_val)
                        if v_by is not None:
                            by_vals.append(v_by)
                        reached_grp = [s for s in group if s["episode_length"] >= turn]
                        n_at = len(reached_grp)
                        c_at = sum(1 for s in reached_grp if s["combo_passes"][combo_name][turn - 1])
                        v_at = _pass_at_k(n_at, c_at, k_val)
                        if v_at is not None:
                            at_vals.append(v_at)
                    pass_k_row[f"pass_at_{k_val}_by_turn"] = sum(by_vals) / len(by_vals) if by_vals else None
                    pass_k_row[f"pass_at_{k_val}_at_turn"] = sum(at_vals) / len(at_vals) if at_vals else None

            row = {
                "combo": combo_name,
                "turn": turn,
                "episodes_total": total_episodes,
                "episodes_reached": episodes_reached,
                "reached_rate": (episodes_reached / total_episodes) if total_episodes else 0.0,
                "successes_at_turn": at_successes,
                "success_rate_at_turn": (at_successes / total_episodes) if total_episodes else 0.0,
                "success_rate_at_turn_reached": (at_successes / episodes_reached) if episodes_reached else None,
                "first_successes_at_turn": first_successes,
                "first_success_rate_at_turn": (first_successes / total_episodes) if total_episodes else 0.0,
                "successes_by_turn": by_successes,
                "success_rate_by_turn": (by_successes / total_episodes) if total_episodes else 0.0,
                **pass_k_row,
            }
            combo_rows.append(row)
            turn_rows.append(row)

        success_lengths = [
            s["first_success_turn"][combo_name]
            for s in episode_summaries
            if s["first_success_turn"][combo_name] is not None
        ]
        combo_summary[combo_name] = {
            "first_success_histogram": {str(k): v for k, v in sorted(first_success_hist.items())},
            "never_succeeded": never_succeeded,
            "turn_rows": combo_rows,
            "success_episode_length": {
                "count": len(success_lengths),
                "mean": round(sum(success_lengths) / len(success_lengths), 3) if success_lengths else None,
                "min": min(success_lengths) if success_lengths else None,
                "max": max(success_lengths) if success_lengths else None,
            },
        }

        complexity_rows = []
        for complexity in ["simple", "complex", "unknown"]:
            group = [s for s in episode_summaries if s["complexity"] == complexity]
            if not group:
                continue
            successes = sum(s["first_success_turn"][combo_name] is not None for s in group)
            never = len(group) - successes
            cx_success_lengths = [
                s["first_success_turn"][combo_name]
                for s in group
                if s["first_success_turn"][combo_name] is not None
            ]
            complexity_rows.append(
                {
                    "combo": combo_name,
                    "complexity": complexity,
                    "episodes_total": len(group),
                    "successes_by_end": successes,
                    "success_rate_by_end": successes / len(group),
                    "never_succeeded": never,
                    "never_succeeded_rate": never / len(group),
                    "success_episode_length_mean": round(sum(cx_success_lengths) / len(cx_success_lengths), 3) if cx_success_lengths else None,
                    "success_episode_length_min": min(cx_success_lengths) if cx_success_lengths else None,
                    "success_episode_length_max": max(cx_success_lengths) if cx_success_lengths else None,
                }
            )
        complexity_summary[combo_name] = complexity_rows

    _avg_prompt_chars: Dict[str, float] = {}
    if episode_summaries:
        _avg_prompt_chars["all"] = sum(s["prompt_chars"] for s in episode_summaries) / len(episode_summaries)
        for _cx in ["simple", "complex", "unknown"]:
            _cx_grp = [s for s in episode_summaries if s["complexity"] == _cx]
            if _cx_grp:
                _avg_prompt_chars[_cx] = sum(s["prompt_chars"] for s in _cx_grp) / len(_cx_grp)

    return {
        "total_episodes": total_episodes,
        "max_turn_observed": max_turn_observed,
        "combo_summary": combo_summary,
        "complexity_summary": complexity_summary,
        "turn_rows": turn_rows,
        "avg_prompt_chars": _avg_prompt_chars,
        "complexity_turn_rows": _build_complexity_turn_rows(
            episode_summaries, pass_k_values, max_turn_observed
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "combo",
        "turn",
        "episodes_total",
        "episodes_reached",
        "reached_rate",
        "successes_at_turn",
        "success_rate_at_turn",
        "success_rate_at_turn_reached",
        "first_successes_at_turn",
        "first_success_rate_at_turn",
        "successes_by_turn",
        "success_rate_by_turn",
    ]
    pass_k_fieldnames: List[str] = []
    if rows:
        by_keys = sorted(
            [k for k in rows[0] if k.startswith("pass_at_") and k.endswith("_by_turn")],
            key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0,
        )
        at_keys = sorted(
            [k for k in rows[0] if k.startswith("pass_at_") and k.endswith("_at_turn")],
            key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0,
        )
        pass_k_fieldnames = by_keys + at_keys
    fieldnames = base_fieldnames + pass_k_fieldnames
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_complexity_csv(path: Path, summary: dict) -> None:
    rows = []
    for combo_name in COMBO_ORDER:
        rows.extend(summary.get("complexity_summary", {}).get(combo_name, []))

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "combo",
        "complexity",
        "episodes_total",
        "successes_by_end",
        "success_rate_by_end",
        "never_succeeded",
        "never_succeeded_rate",
        "success_episode_length_mean",
        "success_episode_length_min",
        "success_episode_length_max",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_complexity_turn_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "complexity",
        "combo",
        "turn",
        "episodes_total",
        "episodes_reached",
        "reached_rate",
        "success_rate_at_turn",
        "success_rate_by_turn",
    ]
    by_keys = sorted(
        [k for k in rows[0] if k.startswith("pass_at_") and k.endswith("_by_turn")],
        key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0,
    )
    at_keys = sorted(
        [k for k in rows[0] if k.startswith("pass_at_") and k.endswith("_at_turn")],
        key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0,
    )
    fieldnames = base_fieldnames + by_keys + at_keys
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_number(value: Optional[float]) -> str:
    return f"{value:.6g}" if value is not None else "n/a"


def write_markdown(path: Path, summary: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Turn Success Summary",
        "",
        f"- Case: `{metadata.get('case_label', '')}`",
        f"- Split: `{metadata.get('split', '')}`",
        f"- Eval reward: `{metadata.get('eval_reward_label', '')}`",
        f"- History length: `{metadata.get('history_length', '')}`",
        f"- Configured max steps: `{metadata.get('max_steps', '')}`",
        f"- Episodes analyzed: `{summary['total_episodes']}`",
        f"- Max turn observed: `{summary['max_turn_observed']}`",
        f"- Avg prompt length (chars): `{summary.get('avg_prompt_chars', {}).get('all', 'n/a'):.0f}`" if isinstance(summary.get("avg_prompt_chars", {}).get("all"), float) else f"- Avg prompt length (chars): `n/a`",
        "",
        "## Complexity Split Summary",
        "",
        "| Combo | Complexity | Episodes | Success by end | Never succeeded | Succ len mean | Succ len min | Succ len max |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for combo_name in COMBO_ORDER:
        for row in summary.get("complexity_summary", {}).get(combo_name, []):
            lines.append(
                f"| {combo_name} | {row['complexity']} | {row['episodes_total']} | "
                f"{_pct(row['success_rate_by_end'])} | {_pct(row['never_succeeded_rate'])} | "
                f"{_fmt_number(row.get('success_episode_length_mean'))} | "
                f"{_fmt_number(row.get('success_episode_length_min'))} | "
                f"{_fmt_number(row.get('success_episode_length_max'))} |"
            )
    lines.append("")


    for combo_name in COMBO_ORDER:
        combo = summary["combo_summary"][combo_name]
        # Detect pass@k columns present in this combo's turn rows
        _pass_by_keys: List[str] = []
        _pass_at_keys: List[str] = []
        if combo["turn_rows"]:
            _first = combo["turn_rows"][0]
            _pass_by_keys = sorted(
                [k for k in _first if k.startswith("pass_at_") and k.endswith("_by_turn")],
                key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0,
            )
            _pass_at_keys = sorted(
                [k for k in _first if k.startswith("pass_at_") and k.endswith("_at_turn")],
                key=lambda x: int(x.split("_")[2]) if x.split("_")[2].isdigit() else 0,
            )
        _pass_headers = [f"pass@{k.split('_')[2]} by turn" for k in _pass_by_keys] + [
            f"pass@{k.split('_')[2]} at turn" for k in _pass_at_keys
        ]
        _all_pass_keys = _pass_by_keys + _pass_at_keys

        header_cols = ["Turn", "Reached", "New success @ turn", "State success among reached", "Success by turn"] + _pass_headers
        sep_cols = [" ---:"] * len(header_cols)
        sel = combo.get("success_episode_length", {})
        sel_str = (
            f"mean=`{sel['mean']:.2f}`, min=`{sel['min']}`, max=`{sel['max']}` (n={sel['count']})"
            if sel.get("count") else "n/a (no successes)"
        )
        lines.extend(
            [
                f"## {combo_name}",
                "",
                f"- Never succeeded: `{combo['never_succeeded']}`",
                f"- First success histogram: `{combo['first_success_histogram']}`",
                f"- Success episode length (turns to first success): {sel_str}",
                "- Invalid actions are failures for their own turn, even if metrics were inherited from the previous state.",
                "",
                "| " + " | ".join(header_cols) + " |",
                "|" + "|".join(sep_cols) + "|",
            ]
        )
        for row in combo["turn_rows"]:
            at_rate = row.get("success_rate_at_turn_reached")
            at_display = _pct(at_rate) if at_rate is not None else "n/a"
            pass_k_cells = [
                _pct(row[k]) if row.get(k) is not None else "n/a"
                for k in _all_pass_keys
            ]
            cells = [
                str(row["turn"]),
                _pct(row["reached_rate"]),
                _pct(row["first_success_rate_at_turn"]),
                at_display,
                _pct(row["success_rate_by_turn"]),
            ] + pass_k_cells
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    data_path = Path(args.data_path) if args.data_path else None
    if data_path is None and args.dataset_dir and args.split:
        data_path = Path(args.dataset_dir) / f"{args.split}.parquet"
    records = load_batch_records(rollout_dir)
    episodes = split_into_episodes(records)
    dataset_metadata = extract_parquet_complexity_metadata(data_path)
    pass_k_values = [int(k) for k in args.pass_k_values.split(",") if k.strip().isdigit()]
    summary = aggregate_episode_stats(episodes, dataset_metadata, pass_k_values=pass_k_values)

    metadata = {
        "case_label": args.case_label,
        "split": args.split,
        "eval_reward_label": args.eval_reward_label,
        "history_length": args.history_length,
        "max_steps": args.max_steps,
        "rollout_dir": str(rollout_dir),
        "data_path": str(data_path) if data_path else "",
    }
    _summary_for_json = {k: v for k, v in summary.items() if k != "complexity_turn_rows"}
    payload = {
        "metadata": {**metadata, "avg_prompt_chars": summary.get("avg_prompt_chars", {})},
        "summary": _summary_for_json,
    }

    write_json(Path(args.output_json), payload)
    write_csv(Path(args.output_csv), summary["turn_rows"])
    complexity_csv = Path(args.output_csv).with_name("complexity_success_summary.csv")
    write_complexity_csv(complexity_csv, summary)
    cx_turn_csv = Path(args.output_csv).with_name("turn_success_by_complexity.csv")
    write_complexity_turn_csv(cx_turn_csv, summary["complexity_turn_rows"])
    write_markdown(Path(args.output_markdown), summary, metadata)

    print(f"[Turn Success] Episodes analyzed: {summary['total_episodes']}")
    print(f"[Turn Success] Max turn observed: {summary['max_turn_observed']}")
    avg_pc = summary.get("avg_prompt_chars", {}).get("all")
    if avg_pc:
        print(f"[Turn Success] Avg prompt chars (all): {avg_pc:.0f}")
    if data_path:
        print(f"[Turn Success] Complexity source: {data_path}")
    print(f"[Turn Success] JSON: {args.output_json}")
    print(f"[Turn Success] CSV: {args.output_csv}")
    print(f"[Turn Success] Complexity CSV: {complexity_csv}")
    print(f"[Turn Success] Complexity turn CSV: {cx_turn_csv}")
    print(f"[Turn Success] Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
