#!/usr/bin/env python3
"""Eval statistics from multi-rollout (pass@k-style) eval dumps.

For a rollout dir with n > 1 rollouts per problem, four-spec success within a turn cutoff (recorded
performance_success on valid steps):

  - pass@1 point estimate (mean over rollouts per problem)
  - 95% CI via problem-level bootstrap
  - simulated single-draw (n=1) distribution: the run-to-run noise of a single-sample eval

Episodes are grouped positionally by --rollouts-per-problem when given, otherwise by goal tuple (tasks with
identical goals merge, slightly widening the CI).

Usage:
  python -m tools.analysis.eval_ci --rollout-dir <dir> [--max-turn 8]
    [--rollouts-per-problem 32] [--bootstrap 4000] [--seed 7]
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st


def load_episodes(rollout_dir: str) -> list[list[dict]]:
    files = sorted(glob.glob(f"{rollout_dir}/batch_*.jsonl"),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    records = [json.loads(l) for f in files for l in open(f) if l.strip()]
    eps, pos = [], 0
    while pos < len(records):
        length = max(1, int(records[pos].get("episode_length", 1)))
        ep = records[pos:pos + length]
        if ep and not ep[0].get("is_padding_sample", 0):
            eps.append(ep)
        pos += length
    return eps


def goal_key(r: dict) -> tuple:
    return (round(float(r.get("goal_gain_min") or 0), 1),
            round(float(r.get("goal_power_max") or 0), 1),
            round(float(r.get("goal_bandwidth_min") or 0), 1),
            str(r.get("goal_phase_margin_min_deg")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--max-turn", type=int, default=8)
    ap.add_argument("--rollouts-per-problem", type=int, default=None,
                    help="if set, group episodes positionally in chunks of this size")
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    eps = load_episodes(args.rollout_dir)

    def success(ep):
        return any(s.get("performance_success") and s.get("is_action_valid", 1) not in (0, False)
                   for s in ep[:args.max_turn])

    if args.rollouts_per_problem:
        k = args.rollouts_per_problem
        groups = [eps[i:i + k] for i in range(0, len(eps), k)]
        counts = [(sum(success(e) for e in g), len(g)) for g in groups]
        grouping = f"positional chunks of {k}"
    else:
        probs: dict[tuple, list[bool]] = {}
        for ep in eps:
            probs.setdefault(goal_key(ep[0]), []).append(success(ep))
        counts = [(sum(v), len(v)) for v in probs.values()]
        grouping = "goal-tuple (identical goals merge — conservative)"
    n_prob = len(counts)
    pass1 = sum(c / l for c, l in counts) / n_prob

    boots, sims = [], []
    for _ in range(args.bootstrap):
        s = [counts[random.randrange(n_prob)] for _ in range(n_prob)]
        boots.append(sum(c / l for c, l in s) / n_prob)
        sims.append(sum(1 for c, l in counts if random.randrange(l) < c) / n_prob)
    boots.sort(); sims.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]

    print(f"dir       : {args.rollout_dir}")
    print(f"episodes  : {len(eps)}  problems: {n_prob}  grouping: {grouping}")
    print(f"pass@1 (all-4 cum@t{args.max_turn}) : {100*pass1:.1f}%")
    print(f"95% CI (problem bootstrap)          : [{100*q(boots,.025):.1f}, {100*q(boots,.975):.1f}]")
    print(f"simulated n=1 eval: sd {100*st.stdev(sims):.1f}pp, 95% range "
          f"[{100*q(sims,.025):.1f}, {100*q(sims,.975):.1f}]")


if __name__ == "__main__":
    main()
