#!/usr/bin/env python3
"""Success vs. turn budget and pass@k from `slic eval` rollout dumps.

    python -m tools.analysis.passk_turns --out figs/passk_turns \\
        --iid "SLIC-Random=outputs/eval/slic_random_standard_iid/rollouts" \\
        --iid "SFT=outputs/eval/sft_standard_iid/rollouts" \\
        --ood "SLIC-Random=outputs/eval/slic_random_standard_ood/rollouts" \\
        [--passk-iid "SLIC-Random=outputs/eval/slic_random_iid_n8/rollouts"] [--passk-ood ...] \\
        [--max-turn 8] [--reported-turn 8] [--annotate best|all|none]

Panels (one column per split that has runs):
  * success within t turns: per sample index (`prompt_sample_index`) the share of tasks whose first all-four-spec success turn is
    <= t; line = mean over the samples, band = std over the samples. `--max-turn` defaults to the largest turn seen in the dumps
    (so an evaluation run with `max_turns: 32` gives the curve to saturation).
  * pass@k (only when --passk-* runs are given): unbiased estimator over the n samples per task of that run, k = 1..n.
Labels are shared between the two rows; a label listed only in one of them is simply absent from the other panel.
Writes <out>.png / .pdf and <out>.json (every plotted number, with the run directories it came from).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.analysis._common import (MARKERS, PALETTE, first_success, legend_below, max_turn_seen, paper_axes, parse_runs,  # noqa: E402
                                    pass_at_k, turn_curve)
from tools.analysis._annotate import annotate_values, mark_x  # noqa: E402

SPLIT_TITLE = {"iid": "In-Distribution", "ood": "Out-of-Distribution"}


def best_curve(pts, i_end, i_rep):
    return max(pts, key=lambda p: (round(float(p[0][i_end]), 6), float(p[0][i_rep])))


def annotate(ax, pts, x_rep, xmax, mode, label_fs=9.5):
    """pts: (y over x = 1..xmax, colour, label). Labels at x = 1 (left), x_rep (right of its line) and xmax (right margin)."""
    if mode == "none" or not pts:
        return
    if x_rep is not None and x_rep >= xmax:
        x_rep = None            # the reference turn is the end of the axis: the end label covers it
    if x_rep is not None:
        mark_x(ax, x_rep, f"turn {x_rep}", dx=0.3)
    if mode == "best":
        pts = [best_curve(pts, xmax - 1, (x_rep or 1) - 1)]
    for x, side, pad in ((1, "left", 0.5), (x_rep, "right", 0.9), (xmax, "right", 0.5)):
        if x is None:
            continue
        annotate_values(ax, x, [(float(y[x - 1]), c) for y, c, _ in pts], side, xpad=pad, min_gap=5.0, fontsize=label_fs,
                        ylim=(0, 94.5 if x == x_rep else 100), yoff=-12.0 if x == x_rep and x != 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iid", action="append", default=[], metavar="LABEL=ROLLOUT_DIR", help="in-distribution run (repeatable)")
    ap.add_argument("--ood", action="append", default=[], metavar="LABEL=ROLLOUT_DIR", help="out-of-distribution run (repeatable)")
    ap.add_argument("--passk-iid", action="append", default=[], metavar="LABEL=ROLLOUT_DIR", help="n-sample run for the IID pass@k panel")
    ap.add_argument("--passk-ood", action="append", default=[], metavar="LABEL=ROLLOUT_DIR", help="n-sample run for the OOD pass@k panel")
    ap.add_argument("--max-turn", type=int, default=0, help="x range of the turn panels (default: largest turn seen)")
    ap.add_argument("--reported-turn", type=int, default=8, help="dotted reference line / label column (0 = none)")
    ap.add_argument("--annotate", choices=("best", "all", "none"), default="best")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    turns = {s: parse_runs(v) for s, v in (("iid", a.iid), ("ood", a.ood)) if v}
    passk = {s: parse_runs(v) for s, v in (("iid", a.passk_iid), ("ood", a.passk_ood)) if v}
    if not turns and not passk:
        raise SystemExit("give at least one --iid/--ood (turn panels) or --passk-iid/--passk-ood (pass@k panels) run")
    labels = []
    for runs in list(turns.values()) + list(passk.values()):
        for lab, _ in runs:
            if lab not in labels:
                labels.append(lab)
    colour = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}
    marker = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(labels)}
    splits = [s for s in ("iid", "ood") if s in turns or s in passk]
    n_rows = int(bool(turns)) + int(bool(passk))
    fig, axes = plt.subplots(n_rows, len(splits), figsize=(6.5 * len(splits), 4.8 * n_rows), squeeze=False, sharey=True)
    data = {"turns": {}, "passk": {}}
    row = 0
    if turns:
        firsts = {s: [(lab, d, first_success(d)) for lab, d in runs] for s, runs in turns.items()}
        xmax = a.max_turn or max(max_turn_seen(f) for fs in firsts.values() for _, _, f in fs)
        x_rep = a.reported_turn or None
        for col, s in enumerate(splits):
            ax, pts = axes[row, col], []
            for k, (lab, d, first) in enumerate(firsts.get(s, [])):
                per, n = turn_curve(first, xmax)
                mean, sd = 100 * per.mean(0), (100 * per.std(0, ddof=1) if per.shape[0] > 1 else np.zeros(xmax))
                x = np.arange(1, xmax + 1)
                ax.plot(x, mean, color=colour[lab], lw=1.8, label=lab, marker=marker[lab], ms=4.5, markevery=max(1, xmax // 8),
                        markeredgecolor="white", markeredgewidth=0.6, zorder=3)
                ax.fill_between(x, mean - sd, mean + sd, color=colour[lab], alpha=0.13, lw=0)
                data["turns"][f"{s}:{lab}"] = {"rollout_dir": d, "tasks": n, "samples": int(per.shape[0]), "mean": mean.round(2).tolist(),
                                               "std": sd.round(2).tolist()}
                pts.append((mean, colour[lab], lab))
                print(f"{s} {lab}: tasks={n} samples={per.shape[0]} success within t=1/{min(4, xmax)}/{xmax}: "
                      f"{mean[0]:.1f} / {mean[min(4, xmax) - 1]:.1f} / {mean[-1]:.1f} %")
            ticks = [1] + [t for t in (4, 8, 12, 16, 20, 24, 28, 32) if t <= xmax] if xmax > 8 else list(range(1, xmax + 1))
            pad = 4.0 if (a.annotate != "none" and xmax > 8) else (0.6 if a.annotate != "none" else 0.0)
            paper_axes(ax, f"{SPLIT_TITLE[s]} — success within t turns", "Turn", 1 - pad, xmax + pad, ticks,
                       "Cumulative pass@1 (%)" if col == 0 else None)
            annotate(ax, pts, x_rep, xmax, a.annotate)
        row += 1
    if passk:
        for col, s in enumerate(splits):
            ax, pts, nmax = axes[row, col], [], 1
            for k, (lab, d) in enumerate(passk.get(s, [])):
                vals, n_tasks, n = pass_at_k(first_success(d))
                y = 100 * np.array(vals); nmax = max(nmax, n)
                ax.plot(np.arange(1, n + 1), y, color=colour[lab], lw=1.8, label=lab, marker=marker[lab], ms=4.5,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=3)
                data["passk"][f"{s}:{lab}"] = {"rollout_dir": d, "tasks": n_tasks, "n": n, "pass_at_k": y.round(2).tolist()}
                pts.append((y, colour[lab], lab))
                print(f"{s} {lab}: pass@k over n={n} samples ({n_tasks} tasks): k=1 {y[0]:.1f} %, k={n} {y[-1]:.1f} %")
            ticks = list(range(1, nmax + 1)) if nmax <= 8 else [1] + [t for t in (4, 8, 12, 16, 20, 24, 28, 32) if t <= nmax]
            pad = 4.0 if (a.annotate != "none" and nmax > 8) else (0.6 if a.annotate != "none" else 0.0)
            paper_axes(ax, f"{SPLIT_TITLE[s]} — pass@k", "k", 1 - pad, nmax + pad, ticks, "Pass@k (%)" if col == 0 else None)
            annotate(ax, pts, None, nmax, a.annotate)
    legend_below(fig, list(axes.flat))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight", facecolor="white")
    Path(f"{out}.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {out}.png / .pdf / .json")


if __name__ == "__main__":
    main()
