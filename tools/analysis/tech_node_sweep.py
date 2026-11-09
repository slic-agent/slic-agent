#!/usr/bin/env python3
"""Grouped bar chart of four-spec pass@1 per technology node (or any other grouping), from `slic eval` rollout dumps.

    python -m tools.analysis.tech_node_sweep --out figs/tech_node_sweep \\
        --run "SLIC-Random=65nm=iid=outputs/eval/slic_random_65nm_iid/rollouts" \\
        --run "SLIC-Random=90nm=iid=outputs/eval/slic_random_90nm_iid/rollouts" \\
        --run "SFT=65nm=iid=outputs/eval/sft_65nm_iid/rollouts" ... \\
        [--nodes 22nm,32nm,45nm,65nm,90nm,130nm] [--models "SFT,SLIC-Random"] [--separate-group 65nm-real]

Each --run is `model=group=split=rollout_dir`. Bar = pass@1 at the run's full turn budget = mean over the sample indices of the
share of tasks solved (all four specs met at some turn); thin error bar = std over the sample indices; the value is printed on
the bar. Missing (model, group) cells are left empty and marked "n/a". One panel per split present (iid / ood).
`--separate-group` draws that group right of a dashed divider with a shaded band (e.g. a real-PDK column next to the
predictive-model nodes). Writes <out>.png / .pdf and <out>.json with every plotted number.

The repository ships the 65 nm predictive model card; `slic data --cards` fetches the 22 / 32 / 45 / 90 / 130 nm cards and
`slic data --family nodes` the standard validation / OOD tasks rebuilt at each node; `configs/eval/node_{validation,ood}.yaml` run
one node (`tech_node` and `tasks` set together; docs/DATASETS.md). The 65 nm bars are the standard validation / test_ood runs.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.analysis._common import PALETTE, first_success, parse_runs  # noqa: E402

HATCHES = ["", "///", "xx", "++", "...", "\\\\", "oo", "--"]
SPLIT_TITLE = {"iid": "In-Distribution", "ood": "Out-of-Distribution"}


def pass1(first):
    samples = sorted({s for v in first.values() for s in v})
    tasks = [t for t, v in first.items() if all(s in v for s in samples)]
    rounds = [np.mean([first[t][s] is not None for t in tasks]) for s in samples]
    return 100 * float(np.mean(rounds)), (100 * float(np.std(rounds, ddof=1)) if len(rounds) > 1 else 0.0), len(tasks), len(samples)


def draw_panel(ax, cells, split, models, groups, separate, colour, hatch, show_ylabel):
    n = len(models); w = min(0.17, 0.85 / n)
    x = np.arange(len(groups), dtype=float)
    if separate in groups:
        x[groups.index(separate):] += 0.35
    for i, m in enumerate(models):
        off = x + (i - (n - 1) / 2) * w
        vals = [cells.get((m, g, split)) for g in groups]
        hs = [v["pass1"] if v else 0.0 for v in vals]; es = [v["std"] if v else 0.0 for v in vals]
        ax.bar(off, hs, w, label=m, color=colour[m], alpha=0.65, hatch=hatch[m], edgecolor="black", linewidth=0.4, zorder=3)
        ok = [v is not None for v in vals]
        ax.errorbar(off[ok], np.array(hs)[ok], yerr=np.array(es)[ok], fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5, zorder=4)
        for o, v, e in zip(off, vals, es):
            if v is None:
                ax.text(o, 0.5, "n/a", ha="center", va="bottom", fontsize=7.5, color="#777777", rotation=90)
            else:
                ax.text(o, min(v["pass1"] + e, 98) + 0.6, f"{v['pass1']:.1f}", ha="center", va="bottom", fontsize=8.5, color="#222222",
                        rotation=90, clip_on=False)
    if separate in groups:
        j = groups.index(separate)
        ax.axvline((x[j - 1] + x[j]) / 2 if j else x[j] - 0.5, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvspan(x[j] - 0.5, x[j] + 0.5, color="#f2e6c8", alpha=0.55, zorder=0, lw=0)
    ax.set_title(SPLIT_TITLE.get(split, split), fontsize=15, fontweight="bold", pad=8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=12)
    ax.set_xlabel("Technology node", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("4-spec pass@1 (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="MODEL=GROUP=SPLIT=ROLLOUT_DIR")
    ap.add_argument("--nodes", default=None, help="comma list: group order on the x axis (default: order of first appearance)")
    ap.add_argument("--models", default=None, help="comma list: bar order inside a group (default: order of first appearance)")
    ap.add_argument("--separate-group", default=None, help="draw this group apart, right of a divider")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args()
    runs = parse_runs(a.run, 4)
    cells, groups, models, splits = {}, [], [], []
    for m, g, s, d in runs:
        p1, sd, n_tasks, n_samples = pass1(first_success(d))
        cells[(m, g, s)] = {"pass1": round(p1, 2), "std": round(sd, 2), "tasks": n_tasks, "samples": n_samples, "rollout_dir": d}
        print(f"{m} @ {g} [{s}]: pass@1 {p1:.1f} ± {sd:.1f} % ({n_tasks} tasks x {n_samples} samples)")
        for lst, v in ((groups, g), (models, m), (splits, s)):
            if v not in lst:
                lst.append(v)
    if a.nodes:
        groups = [g for g in a.nodes.split(",") if g]
    if a.models:
        models = [m for m in a.models.split(",") if m]
    if a.separate_group and a.separate_group in groups:
        groups = [g for g in groups if g != a.separate_group] + [a.separate_group]
    splits = [s for s in ("iid", "ood") if s in splits] + [s for s in splits if s not in ("iid", "ood")]
    colour = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    hatch = {m: HATCHES[i % len(HATCHES)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(splits), figsize=(max(7, 1.6 * len(groups) * max(1, len(models)) / 2) * len(splits), 5.6), squeeze=False, sharey=True)
    for i, s in enumerate(splits):
        draw_panel(axes[0, i], cells, s, models, groups, a.separate_group, colour, hatch, show_ylabel=(i == 0))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour[m], alpha=0.65, hatch=hatch[m], edgecolor="black", linewidth=0.4) for m in models]
    fig.legend(handles, models, loc="lower center", ncol=min(len(models), 5), fontsize=11, frameon=True, framealpha=0.95, edgecolor="#cccccc",
               handlelength=1.6, handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150, bbox_inches="tight", facecolor="white")
    Path(f"{out}.json").write_text(json.dumps({"groups": groups, "models": models, "splits": splits,
                                               "cells": {f"{m}|{g}|{s}": v for (m, g, s), v in cells.items()}}, indent=1))
    print(f"wrote {out}.png / .pdf / .json")


if __name__ == "__main__":
    main()
