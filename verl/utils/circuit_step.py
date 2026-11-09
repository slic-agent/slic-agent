"""Distinguish failed actions from measured, potentially recoverable circuit states."""

from numbers import Integral


def canonical_dump_turn_indices(indices, expected_rows):
    if indices is None or len(indices) != expected_rows:
        raise ValueError('Canonical rollout turn indices must match dump rows.')
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in indices):
        raise ValueError('Canonical rollout turn indices must be nonnegative integers.')
    return [int(value) for value in indices]


def uses_fixed_failure_reward(info):
    if int(info.get("is_action_valid", 1)) == 1:
        return False
    return (info.get("error_type") in {
        "parsing_error", "validation_failed", "incorrect_stop", "geometry_bounds_error", "convergence_failed", "timeout",
        "spectre_crash", "missing_results", "simulation_error", "simulation_parse_error",
    } or info.get("simulation_measurement_valid") is False)
