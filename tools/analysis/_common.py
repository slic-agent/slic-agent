"""Shared loaders for the analysis tools: everything reads the per-turn rollout dumps that `slic eval` writes
(`<output_dir>/rollouts/batch_<i>.jsonl`, one JSON row per turn).

Row fields used here: `task_id`, `prompt_sample_index` (0-based sample / round of the task), `turn_index` (0-based),
`performance_success` (all four specs met at this turn, the environment's verdict), `is_padding_sample`, the metric fields
(`gain_dB`, `power_uW`, `bandwidth_MHz`, ...), the targets (`goal_gain_min`, ...), `topology` (verl/utils/rollout_dump_fields.py).
"""
from __future__ import annotations

import collections
import glob
import json
import os
from math import comb
from pathlib import Path

import numpy as np


def batch_files(rollout_dir: str | os.PathLike) -> list[str]:
    """The batch dumps of one eval run. Accepts the run's output dir (holding `rollouts/`) or the `rollouts/` dir itself."""
    d = Path(rollout_dir)
    if (d / "rollouts").is_dir() and not glob.glob(str(d / "batch_*.jsonl")):
        d = d / "rollouts"
    files = sorted(glob.glob(str(d / "batch_*.jsonl")), key=lambda f: int(Path(f).stem.split("_")[1]))
    if not files:
        raise SystemExit(f"{rollout_dir}: no batch_*.jsonl files")
    return files


def iter_rows(rollout_dir):
    """Every non-padding turn row of the run."""
    for f in batch_files(rollout_dir):
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("is_padding_sample") or r.get("task_id") is None:
                    continue
                yield r


def first_success(rollout_dir) -> dict:
    """task_id -> sample index -> first 1-based turn with all four specs met, or None when the sample never succeeds."""
    first = collections.defaultdict(dict)
    for r in iter_rows(rollout_dir):
        t, s = r["task_id"], int(r.get("prompt_sample_index", 0))
        first[t].setdefault(s, None)
        turn = int(r.get("turn_index", r.get("step", 0))) + 1
        if r.get("performance_success") and (first[t][s] is None or turn < first[t][s]):
            first[t][s] = turn
    return first


def complete_grid(first: dict):
    """Samples seen on every task (tasks missing a sample are dropped) -> (sorted sample ids, task ids)."""
    samples = sorted({s for v in first.values() for s in v})
    tasks = sorted(t for t, v in first.items() if all(s in v for s in samples))
    if not tasks:
        raise SystemExit("no task has every sample index; the dump looks incomplete")
    return samples, tasks


def turn_curve(first: dict, max_turn: int) -> tuple[np.ndarray, int]:
    """Per-sample cumulative success within t turns, t = 1..max_turn: array [samples x max_turn] of task shares, and #tasks."""
    samples, tasks = complete_grid(first)
    per = np.array([[np.mean([first[t][s] is not None and first[t][s] <= k for t in tasks]) for k in range(1, max_turn + 1)]
                    for s in samples])
    return per, len(tasks)


def max_turn_seen(first: dict) -> int:
    turns = [v for d in first.values() for v in d.values() if v is not None]
    return max(turns) if turns else 0


def pass_at_k(first: dict, n: int | None = None) -> tuple[list[float], int, int]:
    """Unbiased pass@k, k = 1..n, over tasks with exactly n samples (n defaults to the common sample count)."""
    if n is None:
        counts = collections.Counter(len(v) for v in first.values())
        n = counts.most_common(1)[0][0]
    tasks = [v for v in first.values() if len(v) == n]
    out = []
    for k in range(1, n + 1):
        vals = []
        for v in tasks:
            c = sum(x is not None for x in v.values())
            vals.append(1 - comb(n - c, k) / comb(n, k) if n - c >= k else 1.0)
        out.append(float(np.mean(vals)))
    return out, len(tasks), n


def parse_runs(items: list[str], n_fields: int = 2) -> list[list[str]]:
    """`label=path` (or `a=b=path`, n_fields parts) command-line entries -> list of parts; the path is the last field."""
    out = []
    for it in items:
        parts = it.split("=", n_fields - 1)
        if len(parts) != n_fields:
            raise SystemExit(f"expected {'='.join(['<f%d>' % i for i in range(1, n_fields)] + ['<rollout_dir>'])}, got {it!r}")
        out.append(parts)
    return out


PALETTE = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def paper_axes(ax, title, xlabel, xmin, xmax, xticks, ylabel=None):
    import matplotlib.pyplot as plt
    ax.set_ylim(0, 100); ax.set_xlim(xmin, xmax); ax.set_xticks(xticks)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, colors="black")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8); sp.set_color("black")
    ax.set_title(title, fontsize=13, fontweight="bold", color="black", pad=8)
    ax.set_xlabel(xlabel, fontsize=12, color="black")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color="black")


def legend_below(fig, axes):
    """One legend under the panels with every label that appears in any of them (first occurrence wins)."""
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    n = len(labels)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, loc="upper center", ncol=n if n <= 5 else (n + 1) // 2, fontsize=10, frameon=True, framealpha=0.95,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.085), handlelength=2.2, handletextpad=0.5, columnspacing=1.4)
