#!/usr/bin/env python3
"""Four-spec success as a function of the spec tolerance, re-judged from `slic eval` rollout dumps (no new inference).

    python -m tools.analysis.tolerance_sweep --out figs/tolerance_sweep \\
        --iid "SLIC-Random=outputs/eval/slic_random_standard_iid/rollouts:outputs/eval/slic_random_standard_iid_tol0/rollouts" \\
        --ood "SLIC-Random=outputs/eval/slic_random_standard_ood/rollouts:outputs/eval/slic_random_standard_ood_tol0/rollouts" \\
        [--tolerances 0-15] [--official 5] [--annotate best|all|none]

Every turn row carries the measured metrics and the four targets (`goal_*`); the specifications, tolerances and validity
gates come from the run's reward preset (`<run>/results.json` next to `<run>/rollouts`, looked up in this repository's
reward_config_example.yaml). The verdict at tolerance t is the environment's own `evaluate_constraint_specs` with the gain /
power / bandwidth / phase-margin relative tolerances all set to t (validity gates unchanged). An episode succeeds at t if any of its turns passes all four specs at t; per sample index the success rate over
tasks, line = mean over samples, band = std. Before sweeping, every row is re-judged at its recorded tolerances and must
reproduce the runner's `performance_success` exactly (fail-fast otherwise).

Caveat: the standard protocol stops an episode at its first success under the official tolerance (5 %), so below it the standard
dumps only give LOWER BOUNDS. For exact values below the official tolerance, run the same checkpoint under the 0 %-tolerance
protocol (`configs/eval/standard_iid_tol0.yaml` / `standard_ood_tol0.yaml`: targets stated at 0 %, auto-stop only at exact
success) and give that run as the second path (`label=standard_dir:tol0_dir`); the points below the official tolerance are then
taken from the 0 % run, the rest from the standard run. Without a second path the figure marks the shaded region as a lower bound.
Writes <out>.png / .pdf, <out>.json and <out>.md (table at 0 / 2 / official / 8 / 10 / 15 %).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "agent_system/environments/env_package/circuitop"))
from circuitop.rewards import evaluate_constraint_specs  # noqa: E402
from reward_config import RewardConfig  # noqa: E402
from tools.analysis._common import MARKERS, PALETTE, iter_rows, parse_runs  # noqa: E402
from tools.analysis._annotate import annotate_values  # noqa: E402

SPECS = ("gain", "power", "bandwidth", "phase_margin")
PRESETS = REPO / "agent_system/environments/env_package/circuitop/configs/reward_config_example.yaml"
METRIC_KEYS = ("gain_dB", "power_uW", "bandwidth_MHz", "bandwidth_kind", "inband_flatness_db", "phase_margin_deg",
               "phase_margin_status", "offset_v", "cmrr_db", "psrr_plus_db", "psrr_minus_db")
TARGETS = {"gain_min": "goal_gain_min", "power_max": "goal_power_max", "bandwidth_min": "goal_bandwidth_min",
           "phase_margin_min_deg": "goal_phase_margin_min_deg", "phase_margin_max_deg": "goal_phase_margin_max_deg"}
TOL_KEYS = ("gain_relative_tolerance", "power_relative_tolerance", "bandwidth_relative_tolerance", "phase_margin_relative_tolerance")
SPLIT_TITLE = {"iid": "In-distribution", "ood": "Out-of-distribution"}


def passes(metrics, goal, tol=None):
    g = dict(goal)
    if tol is not None:
        for k in TOL_KEYS:
            g[k] = tol
    res = evaluate_constraint_specs(metrics, g)
    if not set(SPECS) <= set(res):
        raise SystemExit(f"four-spec verdict incomplete {res} for goal specs {g.get('constraint_specs')}")
    return all(res[s] for s in SPECS)


def run_goal(rollout_dir):
    """The run-constant part of every goal: the preset's specifications, tolerances and validity-gate switch."""
    res = Path(rollout_dir).parent / "results.json"
    if not res.is_file():
        raise SystemExit(f"{res} not found: the sweep needs the run's results.json next to its rollouts directory")
    ref = json.load(open(res))["config"]["env"]["circuitop"]["reward_config"]
    rc = RewardConfig.from_yaml(str(PRESETS), ref.rsplit(":", 1)[1])
    return {"constraint_specs": list(rc.constraint_specs), "disable_validity_gates": rc.disable_validity_gates,
            **{k: getattr(rc, k) for k in TOL_KEYS}}


def sweep(rollout_dir, tols):
    """-> (per-sample success matrix [samples x tols] in %, n_tasks, rows re-judged)."""
    ok = collections.defaultdict(lambda: np.zeros(len(tols), dtype=bool))
    seen, checked = set(), 0
    base = run_goal(rollout_dir)
    for r in iter_rows(rollout_dir):
        key = (r["task_id"], int(r.get("prompt_sample_index", 0))); seen.add(key)
        m = {k: r.get(k) for k in METRIC_KEYS}
        goal = {**base, **{k: r.get(c) for k, c in TARGETS.items()}}
        if passes(m, goal) != bool(r.get("performance_success")):
            raise SystemExit(f"{rollout_dir}: re-judged verdict != runner performance_success for task {key[0]} sample {key[1]} "
                             f"turn {r.get('turn_index')}")
        checked += 1
        ok[key] |= np.array([passes(m, goal, t) for t in tols])
    samples = sorted({s for _, s in seen}); tasks = sorted({t for t, _ in seen})
    if any((t, s) not in seen for t in tasks for s in samples):
        raise SystemExit(f"{rollout_dir}: incomplete task x sample grid")
    per = np.array([[np.mean([ok[(t, s)][i] for t in tasks]) for i in range(len(tols))] for s in samples])
    return 100 * per, len(tasks), checked


def parse_tols(spec):
    if "-" in spec:
        lo, hi = spec.split("-"); return [t / 100 for t in range(int(lo), int(hi) + 1)]
    return [float(t) / 100 for t in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iid", action="append", default=[], metavar="LABEL=STD_DIR[:TOL0_DIR]")
    ap.add_argument("--ood", action="append", default=[], metavar="LABEL=STD_DIR[:TOL0_DIR]")
    ap.add_argument("--tolerances", default="0-15", help="percent, a-b range or comma list (default 0-15)")
    ap.add_argument("--official", type=float, default=5.0, help="tolerance of the standard protocol, percent (default 5)")
    ap.add_argument("--annotate", choices=("best", "all", "none"), default="best", help="value labels at 0 %%, the official tolerance and the end")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    tols = parse_tols(a.tolerances); x = [100 * t for t in tols]
    off = a.official / 100
    if off not in tols:
        raise SystemExit(f"--official {a.official} must be one of the sweep tolerances {a.tolerances}")
    i_off = tols.index(off)
    runs = {s: parse_runs(v) for s, v in (("iid", a.iid), ("ood", a.ood)) if v}
    if not runs:
        raise SystemExit("give at least one --iid or --ood run")
    labels = []
    for rs in runs.values():
        for lab, _ in rs:
            if lab not in labels:
                labels.append(lab)
    colour = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}
    marker = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(labels)}
    splits = list(runs)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.8 * len(splits), 4.4), squeeze=False)
    data = {"tolerances_pct": x, "official_pct": a.official, "curves": {}}
    any_lower_bound = False
    for col, s in enumerate(splits):
        ax, pts = axes[0, col], []
        for lab, paths in runs[s]:
            std_dir, _, tol0_dir = paths.partition(":")
            per, n_tasks, checked = sweep(std_dir, tols)
            src_low = None
            if tol0_dir:
                per0, n0, c0 = sweep(tol0_dir, tols)
                if n0 != n_tasks or per0.shape[0] != per.shape[0]:
                    raise SystemExit(f"{lab}/{s}: 0 %-tolerance run grid {n0}x{per0.shape[0]} != standard run {n_tasks}x{per.shape[0]}")
                per[:, :i_off] = per0[:, :i_off]; checked += c0; src_low = tol0_dir
            else:
                any_lower_bound = True
            mean = per.mean(0); sd = per.std(0, ddof=1) if per.shape[0] > 1 else np.zeros(len(tols))
            ax.plot(x, mean, color=colour[lab], lw=1.6, marker=marker[lab], ms=3.5, markevery=max(1, len(tols) // 5), zorder=3, label=lab)
            ax.fill_between(x, mean - sd, mean + sd, color=colour[lab], alpha=0.12, lw=0)
            data["curves"][f"{s}:{lab}"] = {"standard_dir": std_dir, "tol0_dir": src_low, "tasks": n_tasks, "samples": int(per.shape[0]),
                                            "rows_rejudged": checked, "mean": mean.round(2).tolist(), "std": sd.round(2).tolist()}
            print(f"{s} {lab}: {n_tasks} tasks x {per.shape[0]} samples, {checked} rows re-judged OK; "
                  f"{x[0]:.0f}%={mean[0]:.1f} {a.official:.0f}%={mean[i_off]:.1f} {x[-1]:.0f}%={mean[-1]:.1f}")
            pts.append((mean, colour[lab], lab))
        pad = 2.0 if a.annotate != "none" else 0.3
        ax.axvspan(x[0] - pad, a.official, color="#f2f0ec", zorder=0, lw=0)
        ax.axvline(a.official, color="gray", ls=":", lw=0.9, alpha=0.8)
        ax.text(a.official - 0.25, 98, f"official {a.official:g}%", fontsize=7.5, color="#6b6660", va="top", ha="right")
        ax.set_title(SPLIT_TITLE[s] + f" ({data['curves'][f'{s}:{runs[s][0][0]}']['tasks']} tasks)", fontsize=10)
        ax.set_xlabel("Spec tolerance (%)", fontsize=10)
        if col == 0:
            ax.set_ylabel("4-spec pass@1 (%)", fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(3)); ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.set_xlim(x[0] - pad, x[-1] + pad); ax.set_ylim(0, 100)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6); ax.tick_params(labelsize=9)
        if a.annotate != "none" and pts:
            if a.annotate == "best":
                pts = [max(pts, key=lambda p: (round(float(p[0][-1]), 6), float(p[0][i_off])))]
            for xi, side in ((0, "left"), (i_off, "right"), (len(tols) - 1, "right")):
                annotate_values(ax, x[xi], [(float(m[xi]), c) for m, c, _ in pts], side, xpad=0.3, min_gap=4.6, ylim=ax.get_ylim())
    handles = [Line2D([0], [0], color=colour[lab], marker=marker[lab], ms=3.5, lw=1.6) for lab in labels]
    fig.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8.5, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.97))
    fig.text(0.5, 0.0, (f"Shaded region (< {a.official:g}%): lower bound where no 0%-tolerance run was given (episodes stop at the first "
                        f"success under the official tolerance)." if any_lower_bound else
                        f"Shaded region (< {a.official:g}%): 0%-tolerance evaluation re-judged at each tolerance; >= {a.official:g}%: "
                        "standard evaluation."), ha="center", fontsize=8, color="#6b6660")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight", facecolor="white")
    Path(f"{out}.json").write_text(json.dumps(data, indent=1))
    cols = [i for i in sorted({0, 2, i_off, 8, 10, len(tols) - 1}) if i < len(tols)]
    md = ["# Four-spec success vs spec tolerance (tools/analysis/tolerance_sweep.py)", "",
          "Mean ± std over sample indices, in %. " + ("Values below the official tolerance are lower bounds unless a 0 %-tolerance run was given."
                                                       if any_lower_bound else "Values below the official tolerance come from the 0 %-tolerance runs."), "",
          "| run | split | " + " | ".join(f"{x[i]:g}%" + (" (official)" if i == i_off else "") for i in cols) + " |",
          "|---|---|" + "---|" * len(cols)]
    for k, v in data["curves"].items():
        s, lab = k.split(":", 1)
        md.append(f"| {lab} | {s.upper()} | " + " | ".join(f"{v['mean'][i]:.1f} ± {v['std'][i]:.1f}" for i in cols) + " |")
    Path(f"{out}.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}.png / .pdf / .json / .md")


if __name__ == "__main__":
    main()
