"""
Reward calculation utilities for CircuitOp environment.
"""
from typing import Dict, Any, Optional
import math
import numpy as np


def compute_performance_score(
    metrics: Dict[str, float], 
    goal: Dict[str, float],
    initial_gain_dB: Optional[float] = None,
    initial_power_uW: Optional[float] = None
) -> float:
    """
    Per-turn performance score (higher is better).

    Constraint mode is tolerance-aligned: gain, power and bandwidth are capped at their accepted tolerance boundaries
    and the gated specs (constraint_specs) are averaged with equal weights, so a measured state scores exactly 1 when
    all gated specs pass (then 1 + the success-slack term) and stays below 1 otherwise, also after float32 casting.
    Optimization mode scores the figure of merit against initial_gain_dB / initial_power_uW.
    """
    gain_dB = metrics.get('gain_dB')
    if gain_dB is None:
        gain_dB = metrics.get('differential_gain_db')
    gain_dB = 0.0 if gain_dB is None else float(gain_dB)

    power_uW = metrics.get('power_uW', 1e9)
    power_uW = 1e9 if power_uW is None else float(power_uW)
    
    mode = goal.get('mode', 'constraint')
    
    if mode == 'optimization':
        # Optimization mode (no hard thresholds): maximise gain and minimise power.
        # Figure of merit: relative linear-gain increase + relative power decrease over the initial design
        # (gain normalised by the final linear gain, so a negative initial gain never divides by a negative value).
        if initial_gain_dB is None or initial_power_uW is None:
            initial_gain_dB = goal.get('initial_gain_dB', 0.0)
            initial_power_uW = goal.get('initial_power_uW', 1.0)
        initial_gain_dB = 0.0 if initial_gain_dB is None else float(initial_gain_dB)
        initial_power_uW = 1.0 if initial_power_uW is None else float(initial_power_uW)
        gain_linear_final = 10 ** (gain_dB / 20)
        gain_linear_initial = 10 ** (initial_gain_dB / 20)
        relative_gain_increase = (gain_linear_final - gain_linear_initial) / max(gain_linear_final, 1e-9)
        relative_power_decrease = (initial_power_uW - power_uW) / max(initial_power_uW, 1e-9)
        return 0.5 * relative_gain_increase + 0.5 * relative_power_decrease

    # Constraint mode: tolerance-aligned score, equal weights over constraint_specs.
    target_gain = goal.get('gain_min', 0.0)
    power_budget = goal.get('power_max', 1e9)
    bandwidth_min = goal.get('bandwidth_min')
    phase_margin_min_deg = goal.get('phase_margin_min_deg')
    phase_margin_max_deg = goal.get('phase_margin_max_deg')

    gain_relative_tolerance = float(goal.get('gain_relative_tolerance', 0.02))
    power_relative_tolerance = float(goal.get('power_relative_tolerance', 0.05))
    bandwidth_relative_tolerance = float(goal.get('bandwidth_relative_tolerance', 0.05))
    phase_margin_relative_tolerance = float(goal.get('phase_margin_relative_tolerance', 0.05))
    tolerances = (gain_relative_tolerance, power_relative_tolerance,
                  bandwidth_relative_tolerance, phase_margin_relative_tolerance)
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in tolerances):
        raise ValueError('Tolerance-aligned scoring requires finite tolerances in [0, 1).')
    canonical_gain = metrics.get('differential_gain_db')
    if canonical_gain is None:
        canonical_gain = metrics.get('gain_dB')
    gain_dB = 0.0 if canonical_gain is None else float(canonical_gain)
    # thresholds relaxed by the tolerances: a spec scores 1 exactly when it passes
    target_gain = float(target_gain) - abs(float(target_gain)) * gain_relative_tolerance
    power_budget = float(power_budget) + abs(float(power_budget)) * power_relative_tolerance
    if bandwidth_min is not None:
        bandwidth_min = float(bandwidth_min) - abs(float(bandwidth_min)) * bandwidth_relative_tolerance
    for name, value in (('gain', target_gain), ('power', power_budget), ('bandwidth', bandwidth_min)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'Tolerance-aligned scoring requires a finite positive {name} threshold.')

    bandwidth_mhz = metrics.get('bandwidth_MHz')   # MHz is the only bandwidth key
    bandwidth_mhz = 0.0 if bandwidth_mhz is None else float(bandwidth_mhz)
    # Gate B (domain validity): in-band peaking above 6 dB means the first pole has collapsed into the pass band, so the
    # gain / bandwidth / phase-margin readings are unreliable and those specs score 0. Power (a DC quantity) is unaffected.
    # A non-finite reading counts as missing.
    _ib = metrics.get('inband_flatness_db')
    if _ib is not None and not math.isfinite(float(_ib)):
        _ib = None
    gates_off = bool(goal.get('disable_validity_gates', False))
    inband_invalid = (not gates_off) and _ib is not None and float(_ib) > 6.0

    def gain_score_value() -> float:
        if inband_invalid:
            return 0.0
        if metrics.get('gain_dB') is None and metrics.get('differential_gain_db') is None:
            return 0.0
        if not math.isfinite(gain_dB):
            return 0.0
        if gain_dB >= target_gain:
            return 1.0
        return 10.0 ** ((gain_dB - target_gain) / 20.0)

    def power_score_value() -> float:
        if metrics.get('power_uW') is None:
            return 0.0
        val = float(power_uW)
        if not math.isfinite(val):
            return 0.0
        budget = float(power_budget)
        if val <= budget:
            return 1.0
        if val <= 0.0:
            return 0.0
        return float(np.clip(1.0 - (val - budget) / max(val, 1e-12), 0.0, 1.0))

    def bandwidth_score_value() -> float:
        if inband_invalid:
            return 0.0
        if bandwidth_min is None or float(bandwidth_min) <= 0.0:
            return 0.0
        if not math.isfinite(bandwidth_mhz):
            return 0.0
        target_bw = float(bandwidth_min)
        if bandwidth_mhz >= target_bw:
            return 1.0
        return max(0.0, bandwidth_mhz / target_bw)

    def spec_score(spec_name: str) -> float:
        if spec_name == 'gain':
            return gain_score_value()
        if spec_name == 'power':
            return power_score_value()
        if spec_name == 'bandwidth':
            return bandwidth_score_value()
        # phase margin and any other spec: pass / fail as 1 / 0
        result = evaluate_constraint_specs(metrics, {**goal, 'constraint_specs': [spec_name]})
        return 1.0 if bool(result.get(spec_name, False)) else 0.0

    gate_specs = goal.get('constraint_specs')
    if gate_specs is None:
        raise RuntimeError("Constraint performance score requires explicit constraint_specs.")
    gate_specs_list = [gate_specs] if isinstance(gate_specs, str) else [str(s) for s in gate_specs]
    gate_specs_unique = list(dict.fromkeys(s for s in gate_specs_list if s))
    if not gate_specs_unique:
        raise RuntimeError("Constraint performance score requires non-empty constraint_specs.")

    aligned_passes = evaluate_constraint_specs(metrics, {**goal, 'constraint_specs': gate_specs_unique})
    if set(aligned_passes) != set(gate_specs_unique):
        raise ValueError('Tolerance-aligned scoring requires a valid target for every gated spec.')

    def success_secondary_term():
        """success_margin_weight * mean slack term over the gated specs (formula in RewardConfig)."""
        w_m = float(goal.get('success_margin_weight', 0.0) or 0.0)
        if w_m < 0.0 or not math.isfinite(w_m):
            raise ValueError('success_margin_weight must be a finite number >= 0.')
        if w_m == 0.0:
            return 0.0
        cap = float(goal.get('success_slack_cap', 1.0))
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError('success_slack_cap must be a positive finite number.')
        nominal_gain = float(goal.get('gain_min', 0.0))
        nominal_power = float(goal.get('power_max', 1e9))
        nominal_bw = goal.get('bandwidth_min')
        tol_of = dict(gain=gain_relative_tolerance, power=power_relative_tolerance, bandwidth=bandwidth_relative_tolerance)
        terms, pm_terms = [], []   # PM term last (fixed summation order)
        for spec_name in gate_specs_unique:
            if spec_name == 'gain':
                if nominal_gain == 0.0:
                    raise ValueError('success slack needs a nonzero gain_min.')
                raw = (float(gain_dB) - nominal_gain) / abs(nominal_gain)
            elif spec_name == 'power':
                raw = (nominal_power - float(power_uW)) / nominal_power
            elif spec_name == 'bandwidth':
                ref = float(nominal_bw) if nominal_bw is not None else 0.0
                bw = metrics.get('bandwidth_MHz')
                if bw is None or ref <= 0.0:
                    raise ValueError('success slack needs a measured bandwidth and a positive bandwidth_min.')
                raw = (float(bw) - ref) / ref
            elif spec_name == 'phase_margin':
                pm = metrics.get('phase_margin_deg')
                if pm is None:
                    # The PM spec passes without a number only for 'stable_no_crossing' (no unity-gain crossing
                    # = unconditionally stable, evaluate_constraint_specs): treat as maximal PM robustness.
                    if metrics.get('phase_margin_status') != 'stable_no_crossing':
                        raise ValueError('success slack: PM spec passed without a measured phase margin or a stable_no_crossing status.')
                    pm_terms.append(1.0)
                    continue
                if phase_margin_min_deg is None or phase_margin_max_deg is None:
                    raise ValueError('success slack needs a PM window (phase_margin_min_deg and phase_margin_max_deg).')
                lo, hi = float(phase_margin_min_deg), float(phase_margin_max_deg)
                if not hi > lo:
                    raise ValueError('success slack needs a proper PM window (max > min).')
                centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * (1.0 + phase_margin_relative_tolerance)
                pm_terms.append(min(max(1.0 - abs(float(pm) - centre) / half, 0.0), 1.0))
                continue
            else:
                raise ValueError(f'success slack: unsupported gated spec {spec_name!r}.')
            terms.append(min(max((raw + tol_of[spec_name]) / (cap + tol_of[spec_name]), 0.0), 1.0))
        terms += pm_terms
        term = w_m * (sum(terms) / len(terms) if terms else 0.0)
        if not math.isfinite(term) or term < 0.0:
            raise FloatingPointError(f'Invalid success secondary term: {term}')
        return term

    def finish_score(score):
        if not math.isfinite(score) or not 0.0 <= score <= 1.0 + 1e-12:
            raise FloatingPointError(f'Invalid tolerance-aligned score: {score}')
        if all(aligned_passes.values()):
            return 1.0 + success_secondary_term()
        ceiling = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
        return min(score, ceiling)

    # Gate A (domain validity): with negative gain only the gain term counts.
    if gain_dB < 0.0 and not gates_off:
        return finish_score(gain_score_value())
    each = 1.0 / float(len(gate_specs_unique))
    return finish_score(sum(each * spec_score(spec_name) for spec_name in gate_specs_unique))


def _resolve_constraint_specs(goal: Dict[str, float]) -> list[str]:
    """Resolve the hard constraint list for constraint-mode success checks."""
    constraint_specs = goal.get('constraint_specs', None)

    if constraint_specs is not None:
        return list(constraint_specs)

    constraint_specs = []
    if 'gain_min' in goal:
        constraint_specs.append('gain')
    if 'power_max' in goal:
        constraint_specs.append('power')
    if goal.get('bandwidth_min') is not None and goal.get('bandwidth_min') > 0:
        constraint_specs.append('bandwidth')
    if goal.get('phase_margin_min_deg') is not None and goal.get('phase_margin_min_deg') > 0:
        constraint_specs.append('phase_margin')
    if goal.get('offset_abs_max_v') is not None:
        constraint_specs.append('offset')
    if goal.get('cmrr_min_db') is not None:
        constraint_specs.append('cmrr')
    if goal.get('psrr_plus_min_db') is not None:
        constraint_specs.append('psrr_plus')
    if goal.get('psrr_minus_min_db') is not None:
        constraint_specs.append('psrr_minus')
    return constraint_specs


def evaluate_constraint_specs(metrics: Dict[str, float], goal: Dict[str, float]) -> Dict[str, bool]:
    """Return pass/fail for each active hard constraint in constraint mode."""
    gain_value = metrics.get('differential_gain_db')
    if gain_value is None:
        gain_value = metrics.get('gain_dB')
    power_uW = metrics.get('power_uW')
    bandwidth_MHz = metrics.get('bandwidth_MHz', 0.0)
    phase_margin_deg = metrics.get('phase_margin_deg')
    offset_v = metrics.get('offset_v')
    cmrr_db = metrics.get('cmrr_db')
    psrr_plus_db = metrics.get('psrr_plus_db')
    psrr_minus_db = metrics.get('psrr_minus_db')

    gain_relative_tolerance = goal.get('gain_relative_tolerance', 0.02)
    power_relative_tolerance = goal.get('power_relative_tolerance', 0.05)
    bandwidth_relative_tolerance = goal.get('bandwidth_relative_tolerance', 0.05)
    phase_margin_relative_tolerance = goal.get('phase_margin_relative_tolerance', 0.05)

    target_gain = goal.get('gain_min', -1e9)
    power_budget = goal.get('power_max', 1e9)
    bandwidth_min = goal.get('bandwidth_min', None)
    phase_margin_min_deg = goal.get('phase_margin_min_deg', None)
    offset_abs_max_v = goal.get('offset_abs_max_v', None)
    cmrr_min_db = goal.get('cmrr_min_db', None)
    psrr_plus_min_db = goal.get('psrr_plus_min_db', None)
    psrr_minus_min_db = goal.get('psrr_minus_min_db', None)

    constraint_specs = _resolve_constraint_specs(goal)
    results: Dict[str, bool] = {}
    # disable_validity_gates: spec verdicts ignore the in-band flatness gate and the positive-gain requirement of the
    # PM no-crossing rule (the no-gates ablation preset; the evaluation presets keep the gates).
    gates_off = bool(goal.get('disable_validity_gates', False))

    if 'gain' in constraint_specs:
        _ibg = None if gates_off else metrics.get('inband_flatness_db')
        if _ibg is not None and float(_ibg) > 6.0:
            results['gain'] = False
        elif gain_value is None or not math.isfinite(float(gain_value)):
            results['gain'] = False
        else:
            gain_tolerance = abs(target_gain) * gain_relative_tolerance
            results['gain'] = float(gain_value) >= (target_gain - gain_tolerance)

    if 'power' in constraint_specs:
        if power_uW is None or not math.isfinite(float(power_uW)):
            results['power'] = False
        else:
            power_tolerance = abs(power_budget) * power_relative_tolerance
            results['power'] = float(power_uW) <= (power_budget + power_tolerance)

    if 'bandwidth' in constraint_specs and bandwidth_min is not None and bandwidth_min > 0:
        bandwidth_tolerance = abs(bandwidth_min) * bandwidth_relative_tolerance
        _ib2 = None if gates_off else metrics.get('inband_flatness_db')
        if bandwidth_MHz is None or not math.isfinite(float(bandwidth_MHz)) or (_ib2 is not None and float(_ib2) > 6.0):
            # no 0 dB crossing (bandwidth None) or in-band peaking above the gate -> not met
            results['bandwidth'] = False
        else:
            results['bandwidth'] = float(bandwidth_MHz) >= (bandwidth_min - bandwidth_tolerance)

    if 'phase_margin' in constraint_specs and phase_margin_min_deg is not None and phase_margin_min_deg > 0:
        _ib3 = metrics.get('inband_flatness_db')
        if gates_off and phase_margin_deg is None:
            results['phase_margin'] = metrics.get('phase_margin_status') == 'stable_no_crossing'
        elif (not gates_off) and _ib3 is not None and float(_ib3) > 6.0:
            results['phase_margin'] = False
        elif phase_margin_deg is None:
            # stable_no_crossing passes only with a clean in-band response and positive gain (for a dead
            # circuit "stable" is vacuous); a missing status fails.
            results['phase_margin'] = (metrics.get('phase_margin_status') == 'stable_no_crossing'
                                       and has_positive_gain(metrics)
                                       and _ib3 is not None and float(_ib3) <= 6.0)
        else:
            pm_val = float(phase_margin_deg)
            pm_ok = math.isfinite(pm_val) and pm_val >= float(phase_margin_min_deg) * (1 - phase_margin_relative_tolerance)
            phase_margin_max_deg = goal.get('phase_margin_max_deg')
            if phase_margin_max_deg is not None:
                pm_ok = pm_ok and pm_val <= float(phase_margin_max_deg) * (1 + phase_margin_relative_tolerance)
            results['phase_margin'] = pm_ok

    if 'offset' in constraint_specs and offset_abs_max_v is not None:
        results['offset'] = offset_v is not None and abs(float(offset_v)) <= float(offset_abs_max_v)

    if 'cmrr' in constraint_specs and cmrr_min_db is not None:
        results['cmrr'] = cmrr_db is not None and float(cmrr_db) >= float(cmrr_min_db)

    if 'psrr_plus' in constraint_specs and psrr_plus_min_db is not None:
        results['psrr_plus'] = psrr_plus_db is not None and float(psrr_plus_db) >= float(psrr_plus_min_db)

    if 'psrr_minus' in constraint_specs and psrr_minus_min_db is not None:
        results['psrr_minus'] = psrr_minus_db is not None and float(psrr_minus_db) >= float(psrr_minus_min_db)

    return results


def has_positive_gain(metrics: Dict[str, float]) -> bool:
    """Return True only when the measured gain is positive (> 0 dB)."""
    gain_value = metrics.get('differential_gain_db')
    if gain_value is None:
        gain_value = metrics.get('gain_dB')
    if gain_value is None:
        return False
    try:
        return math.isfinite(float(gain_value)) and float(gain_value) > 0.0
    except (TypeError, ValueError):
        return False


def apply_positive_gain_gate_to_constraint_results(
    results: Dict[str, bool],
    metrics: Dict[str, float],
    goal: Optional[Dict[str, Any]] = None,
) -> Dict[str, bool]:
    """Gate non-gain reporting metrics by positive gain without changing reward logic (no-op under disable_validity_gates)."""
    if (goal is not None and bool(goal.get('disable_validity_gates', False))) or has_positive_gain(metrics):
        return dict(results)
    return {
        spec: bool(passed) if spec == 'gain' else False
        for spec, passed in results.items()
    }


def check_gain_improved(metrics: Dict[str, float], goal: Dict[str, float], initial_gain_dB: Optional[float] = None) -> bool:
    """Optimization mode: gain improved over the initial design by at least gain_improvement_tolerance_dB."""
    gain_value = metrics.get('differential_gain_db')
    if gain_value is None:
        gain_value = metrics.get('gain_dB')
    if gain_value is None or initial_gain_dB is None:
        return False
    return float(gain_value) - float(initial_gain_dB) >= float(goal.get('gain_improvement_tolerance_dB', 0.1))


def check_core_success(metrics: Dict[str, float], goal: Dict[str, float]) -> bool:
    """Benchmark success: gain, bandwidth, power and the phase-margin window all pass, independent of
    constraint_specs (which define `performance_success`)."""
    core_goal = dict(goal or {})
    core_goal["constraint_specs"] = ["gain", "bandwidth", "power", "phase_margin"]
    results = evaluate_constraint_specs(metrics, core_goal)
    required = ("gain", "bandwidth", "power", "phase_margin")
    return all(bool(results.get(spec, False)) for spec in required)


def check_goal_met(metrics: Dict[str, float], goal: Dict[str, float], initial_gain_dB: Optional[float] = None) -> bool:
    """Constraint mode (every released task): every active constraint spec passes (evaluate_constraint_specs).
    Optimization mode: the gain improved over the initial design (check_gain_improved). False when gain or power is missing."""
    gain_dB = metrics.get('differential_gain_db')
    if gain_dB is None:
        gain_dB = metrics.get('gain_dB')
    power_uW = metrics.get('power_uW')
    
    if gain_dB is None or power_uW is None:
        return False
    
    mode = goal.get('mode', 'constraint')
    
    if mode == 'optimization':
        return check_gain_improved(metrics, goal, initial_gain_dB=initial_gain_dB)
    results = evaluate_constraint_specs(metrics, goal)
    return bool(results) and all(results.values())
