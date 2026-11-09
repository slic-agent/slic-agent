#!/usr/bin/env python3
"""Paired bootstrap significance test between two `slic eval` runs on the SAME task set.

    python -m tools.analysis.paired_bootstrap --run-a outputs/eval/a/rollouts --run-b outputs/eval/b/rollouts \\
        [--label-a A --label-b B] [--turns 4 8] [--n-boot 10000] [--seed 7]

Pairs at the task level (`task_id`): for each task the per-run value is the mean over its samples of "all four specs met within
t turns". Tasks are resampled with replacement B times; reports the mean difference A - B, the percentile 95 % CI and a two-sided
bootstrap p-value, for every turn budget in --turns. Fails fast when the two runs do not cover the same tasks.
"""
from __future__ import annotations

import argparse
import random
import statistics as st

from tools.analysis._common import first_success


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", required=True); ap.add_argument("--run-b", required=True)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--turns", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    A, B = first_success(a.run_a), first_success(a.run_b)
    tasks = sorted(set(A) & set(B))
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        raise SystemExit(f"runs do not cover the same tasks: {len(only_a)} only in A (e.g. {only_a[:3]}), {len(only_b)} only in B "
                         f"(e.g. {only_b[:3]})")
    rng = random.Random(a.seed)
    print(f"paired bootstrap: {a.label_a} vs {a.label_b}  tasks={len(tasks)}  samples per task: "
          f"{st.mean(len(A[t]) for t in tasks):.1f} / {st.mean(len(B[t]) for t in tasks):.1f}  B={a.n_boot}")
    for t in a.turns:
        va = [st.mean(f is not None and f <= t for f in A[task].values()) for task in tasks]
        vb = [st.mean(f is not None and f <= t for f in B[task].values()) for task in tasks]
        d = [x - y for x, y in zip(va, vb)]
        boots = sorted(st.mean(d[rng.randrange(len(d))] for _ in range(len(d))) for _ in range(a.n_boot))
        lo, hi = boots[int(0.025 * a.n_boot)], boots[int(0.975 * a.n_boot)]
        ple = sum(1 for x in boots if x <= 0); pge = sum(1 for x in boots if x >= 0)
        p = min(1.0, 2 * min(ple + 1, pge + 1) / (a.n_boot + 1))
        print(f"  t{t:<2d} n={len(tasks):4d}  {a.label_a}={100 * st.mean(va):5.1f}  {a.label_b}={100 * st.mean(vb):5.1f}  "
              f"diff={100 * st.mean(d):+5.1f} pp  95% CI [{100 * lo:+5.1f}, {100 * hi:+5.1f}]  p={p:.4f}  "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")


if __name__ == "__main__":
    main()
