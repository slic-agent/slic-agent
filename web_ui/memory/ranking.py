"""Nearest-start retrieval: metadata filtering -> constraint-aware weighted distance -> reranking.

Stage 1, metadata filter (web_ui/memory/routes.py supplies it). A stored state is a candidate only when it shares the query's
operating conditions:
  - the same task key (topology | technology node);
  - the same condition values (supply vdd);
  - from a saved run;
  - a valid design.
States that differ in a condition are not comparable designs, so they are never ranked.

Stage 2, constraint-aware distance (the kNN metric). Each spec j turns its goal into a tolerated bound b_j (same formulas as the
environment's verdict):
  - min: b = g - |g| t
  - max: b = g + |g| t
  - window: [g_min (1 - t), g_max (1 + t)]
The violation is one-sided, so exceeding a lower bound or staying under an upper bound costs nothing:
    lower bound:  v_j = max(0, d(b_j, x_j))        upper bound:  v_j = max(0, d(x_j, b_j))
    D(x, q) = sum_j w_j v_j                        (w_j = 1 unless the query gives weights)
d(a, c) is the signed distance from c up to a, measured on one shared scale, "decades":
    scale "log": d = log10(a / c)        for positive quantities spanning orders of magnitude (power, bandwidth, PM)
    scale "db":  d = (a - c) / 20        for quantities already in dB (gain: 20 dB = one decade of voltage gain)
So 0.1 means "a factor 10^0.1 = 1.26 off" for every spec, whatever its unit. Gain in dB and power in uW or mW therefore weigh
the same per relative shortfall. The scale is fixed by physics, not by the data, so a ranking does not shift as the database
grows, and it is defined even with one candidate. Z-score and min-max scaling depend on the current pool: rankings would
change as designs are saved, and outliers would stretch the scale.
A spec whose metric is missing, or non-positive on a log scale (e.g. no -3 dB point, an unstable PM <= 0), cannot be
placed on the scale. It gets UNMEASURABLE_PENALTY decades and is flagged. Such designs stay in the ranking, below every
measurable design with less total violation.

Stage 3, reranking of equal distances:
  1. D ascending;
  2. robustness descending: the smallest per-spec slack (signed, in decades, inside the bound);
  3. similarity: the weighted two-sided distance sqrt(sum_j w_j d(x_j, b_j)^2) to the bound point, ascending (the least
     over-designed among equally robust designs).
Identical states are listed once.

Every candidate keeps its per-spec v_j, slack and value, so a failing design shows which specs it misses and by how much.
D = 0 on the specs is an *estimate* of success when the specs leave out gates the environment applies. When the
stored run used the same goal and tolerances, its recorded verdict is exact and is reported as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

UNMEASURABLE_PENALTY = 1.0   # decades: as if the metric were a factor 10 outside its bound


@dataclass(frozen=True)
class Spec:
    name: str                      # tolerance / weight key, e.g. "gain"
    metric: str                    # metric key in a state's metrics, e.g. "gain_dB"
    kind: str                      # "min" | "max" | "window"
    goal_key: str                  # e.g. "gain_min"; for a window, its lower bound
    goal_max_key: Optional[str] = None   # window upper bound
    positive_goal_only: bool = False
    scale: str = "log"             # "log" (positive quantity, decades of ratio) | "db" (value in dB, /20)

    def __post_init__(self):
        if self.kind not in ("min", "max", "window"):
            raise ValueError(f"spec {self.name}: kind {self.kind!r}")
        if (self.kind == "window") != (self.goal_max_key is not None):
            raise ValueError(f"spec {self.name}: goal_max_key is required for (only) a window")
        if self.scale not in ("log", "db"):
            raise ValueError(f"spec {self.name}: scale {self.scale!r}")


def _finite(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _d(scale: str, a: float, c: float) -> Optional[float]:
    """Signed distance from c up to a in decades; None when a log-scale operand is not positive."""
    if scale == "db":
        return (a - c) / 20.0
    if a <= 0 or c <= 0:
        return None
    return math.log10(a / c)


def evaluate(specs: Sequence[Spec], metrics: Dict[str, Any], goal: Dict[str, Any], tolerances: Dict[str, float]
             ) -> Dict[str, Dict[str, Any]]:
    """Per active spec: {pass, violation (decades, >= 0), slack (decades, signed; None if unmeasurable), measurable, value,
    bound, gap (= slack: the signed distance to the binding bound, used by the similarity term)}. Raises when an active spec has no tolerance."""
    out: Dict[str, Dict[str, Any]] = {}
    for s in specs:
        g = _finite(goal.get(s.goal_key))
        if g is None or (s.positive_goal_only and g <= 0):
            continue
        if s.name not in tolerances or _finite(tolerances[s.name]) is None:
            raise ValueError(f"no tolerance for spec {s.name}")
        t = float(tolerances[s.name])
        x = _finite(metrics.get(s.metric))
        if s.kind == "min":
            bound = {"min": g - abs(g) * t}
        elif s.kind == "max":
            bound = {"max": g + abs(g) * t}
        else:
            gmax = _finite(goal.get(s.goal_max_key))
            bound = {"min": g * (1 - t), "max": None if gmax is None else gmax * (1 + t)}
        slacks = []
        if x is not None:
            if bound.get("min") is not None:
                slacks.append(_d(s.scale, x, bound["min"]))     # >= 0 when x is at or above the lower bound
            if bound.get("max") is not None:
                slacks.append(_d(s.scale, bound["max"], x))     # >= 0 when x is at or below the upper bound
        if x is None or any(v is None for v in slacks):
            out[s.name] = {"pass": False, "violation": UNMEASURABLE_PENALTY, "slack": None, "measurable": False,
                           "value": x, "bound": bound, "gap": None}
            continue
        slack = min(slacks)
        out[s.name] = {"pass": slack >= 0, "violation": max(0.0, -slack), "slack": slack, "measurable": True,
                       "value": x, "bound": bound, "gap": slack}
    return out


def _same(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]], keys: Sequence[str]) -> bool:
    if a is None or b is None:
        return False
    for k in keys:
        x, y = _finite(a.get(k)), _finite(b.get(k))
        if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-12 * max(1.0, abs(x))):
            return False
    return True


def rank(specs: Sequence[Spec], candidates: List[Dict[str, Any]], goal: Dict[str, Any], tolerances: Dict[str, float],
         k: int = 5, weights: Optional[Dict[str, float]] = None,
         condition_filter: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None) -> Dict[str, Any]:
    """Rank store.candidate_states(...) rows for (goal, tolerances). condition_filter(candidate) returns the name of the
    condition it fails, or None. Returns {"candidates": top k, "funnel": counts per stage, "weights": used}."""
    names = [s.name for s in specs]
    w = {n: 1.0 for n in names}
    for n, v in (weights or {}).items():
        if n not in w:
            raise ValueError(f"weight for unknown spec {n!r}; specs are {names}")
        f = _finite(v)
        if f is None or f < 0:
            raise ValueError(f"weight {n}={v!r}: a finite number >= 0")
        w[n] = f
    if not any(w.values()):
        raise ValueError("all weights are 0")
    goal_keys = [s.goal_key for s in specs] + [s.goal_max_key for s in specs if s.goal_max_key]
    funnel = {"pool": len(candidates), "excluded_by_condition": {}, "ranked": 0, "unique": 0}
    seen, scored = set(), []
    for c in candidates:
        why = condition_filter(c) if condition_filter else None
        if why:
            funnel["excluded_by_condition"][why] = funnel["excluded_by_condition"].get(why, 0) + 1
            continue
        ev = evaluate(specs, c["metrics"], goal, tolerances)
        if not ev:
            raise ValueError("goal activates no spec")
        dist = sum(w[n] * e["violation"] for n, e in ev.items())
        meas = [e["slack"] for e in ev.values() if e["measurable"]]
        robust = min(meas) if len(meas) == len(ev) else -math.inf
        sim = math.sqrt(sum(w[n] * e["gap"] ** 2 for n, e in ev.items() if e["measurable"])) if meas else math.inf
        exact = (c["step"] >= 0 and c["success"] is not None and _same(c["run_goal"], goal, goal_keys)
                 and _same(c["run_tolerances"], tolerances, names))
        scored.append((dist, -robust, sim, c, ev, exact))
    funnel["ranked"] = len(scored)
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]["run_created"], x[3]["step"]))
    out = []
    for dist, neg_robust, sim, c, ev, exact in scored:
        if c["state_sha256"] in seen:
            continue
        seen.add(c["state_sha256"])
        unmeas = [n for n, e in ev.items() if not e["measurable"]]
        if exact:
            verdict = "exact_success" if c["success"] else "exact_fail"
        elif unmeas:
            verdict = "unmeasurable"
        else:
            verdict = "estimated_pass" if dist == 0 else "estimated_fail"
        out.append({
            "run_id": c["run_id"], "step": c["step"], "state": c["state"], "metrics": c["metrics"],
            "distance": dist, "robustness": -neg_robust, "similarity": sim, "specs": ev, "unmeasurable": unmeas,
            "verdict": verdict, "run_label": c["run_label"], "run_created": c["run_created"],
        })
    funnel["unique"] = len(seen)
    return {"candidates": out[:k], "funnel": funnel, "weights": w}
