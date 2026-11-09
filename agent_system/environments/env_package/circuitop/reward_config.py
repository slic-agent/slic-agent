"""
Reward configuration and terminal reward for the CircuitOp environment.

RewardConfig holds one preset of configs/reward_config_example.yaml. The per-turn score (perf_score_post, the signal the
turn-level estimators train on) is computed by circuitop.rewards.compute_performance_score; RewardComputer computes the
environment's episode reward: 0 before the done step, the terminal metric bonus at the done step.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


@dataclass
class RewardConfig:
    """One reward preset. Load with RewardConfig.from_yaml(path, name) or load_reward_config('path:name')."""

    # ===== Goal =====
    # Set per task by the environment worker (dataset targets plus the fields below); constraint mode
    # (gain >= X, power <= Y, ...) or optimization mode (maximise gain, minimise power).
    goal: Optional[Dict[str, float]] = None
    # The task's specs, injected into every goal as goal['constraint_specs']. One list for three roles:
    # performance_success (all listed specs met), the stop rule (the episode ends when all pass, unless
    # allow_further_exploration_after_goal) and the per-turn score (mean over these specs). The benchmark `success`
    # metric always checks gain / bandwidth / power / phase margin, independent of this list (rewards.check_core_success).
    constraint_specs: list = field(default_factory=lambda: ['gain', 'power', 'bandwidth', 'phase_margin'])

    # ===== Actions =====
    # True: action percentages must lie in [action_percentage_min, action_percentage_max] (shown in the prompt).
    enforce_action_percentage_range: bool = True
    action_percentage_min: Optional[float] = 1
    action_percentage_max: Optional[float] = 500
    # Prompt display limit for the target list of the admissible actions: None or <= 0 shows all targets.
    action_space_prompt_max_targets: Optional[int] = 0
    # Action encoding: "percentage" (<action>increase <target> <param> <percentage>%</action>) or
    # "absolute" (<action><param>=<value>; ...</action>).
    action_change_mode: str = "percentage"

    # ===== Per-turn score: success slack =====
    # Before success the per-turn score is the tolerance-aligned mean of the per-spec scores over constraint_specs (< 1).
    # Once all gated specs pass it is 1 + success_margin_weight * mean(slack terms):
    #   gain, bandwidth, power: relative slack to the nominal goal ((g - g_min)/|g_min|, (bw - bw_min)/bw_min,
    #     (p_max - p)/p_max) as clip((r + tol) / (success_slack_cap + tol), 0, 1), saturating at success_slack_cap
    #     (1.0 = +100 % headroom);
    #   phase margin: clip(1 - |PM - c| / (h * (1 + tol)), 0, 1), c = window centre, h = half-width; 1 for a design
    #     without a unity-gain crossing.
    # 0 = no slack term (every success scores 1.0).
    success_margin_weight: float = 0.0
    success_slack_cap: float = 1.0

    # ===== Domain-validity gates =====
    # Default on. True removes both, from the per-turn score, the spec verdicts (success included) and the terminal
    # bonus: gate A — power / bandwidth / phase margin only count (and only pass) once the gain is positive, and no
    # terminal bonus is paid without positive gain; gate B — in-band peaking above 6 dB fails gain, bandwidth and
    # phase margin. The simulator's design error for gain <= 0 (an invalid action) is not affected.
    disable_validity_gates: bool = False

    # ===== Terminal metric bonus (the environment's episode reward, paid at the done step) =====
    # terminal_metric_bonus[spec] for each met spec (specs without a target in the goal are skipped), 0 when the final
    # action is invalid.
    terminal_metric_bonus: Dict[str, float] = field(default_factory=lambda: {
        'gain': 5.0,
        'power': 3.0,
        'phase_margin': 1.0,
        'bandwidth': 1.0,
    })
    # True: no bonus at all unless the gain spec is met (at terminal_bonus_gain_relative_tolerance).
    require_gain_met_for_terminal_bonus: bool = True
    # True: phase margin adds its own term when met (otherwise it is not part of the bonus).
    include_phase_margin_in_terminal_bonus: bool = False

    # ===== Optimization-mode goals (goal['mode'] == 'optimization', no hard thresholds) =====
    # The per-turn score is a figure of merit: 0.5 * relative linear-gain increase + 0.5 * relative power decrease over
    # the initial design (rewards.py). The episode reward is paid when the gain improved by gain_improvement_tolerance_dB:
    #   "fixed_bonus": success_bonus;  "performance_score": the figure of merit.
    sparse_optimization_style: str = "fixed_bonus"
    success_bonus: float = 12.0

    # ===== Termination =====
    # An episode ends when every spec in constraint_specs passes (unless allow_further_exploration_after_goal) or at the
    # turn limit.
    # True: goal met -> the episode continues until the turn limit (contopt presets).
    allow_further_exploration_after_goal: bool = False

    # ===== Prompt =====
    # "goal_with_optimization_hint": goal with explicit targets + goal_optimization_hint;
    # "goal_no_hint": goal with explicit targets and no optimisation / priority / recovery text (goal_optimization_hint
    #   must be unset). These are the two released prompts (agent_system/environments/prompts/circuitop.py).
    goal_prompt_style: str = "goal_with_optimization_hint"
    # The optimisation hint of goal_with_optimization_hint (required there).
    goal_optimization_hint: Optional[str] = None

    # ===== Stop action =====
    # <action>success</action> while the goal is not met ends the episode with this environment reward.
    stop_incorrect_penalty: float = -3.0

    # ===== Tolerances =====
    gain_improvement_tolerance_dB: float = 0.1  # Minimum gain improvement in optimization mode (dB)
    gain_relative_tolerance: float = 0.05  # Gain may fall this fraction below its target
    terminal_bonus_gain_relative_tolerance: float = 0.05  # Gain tolerance of the terminal bonus' gain check
    power_relative_tolerance: float = 0.05  # Power may exceed its budget by this fraction
    phase_margin_relative_tolerance: float = 0.05  # PM window [min x (1 - t), max x (1 + t)]
    bandwidth_relative_tolerance: float = 0.05  # Bandwidth may fall this fraction below its target

    def to_dict(self) -> Dict[str, Any]:
        """All fields as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RewardConfig':
        """Create config from dictionary. Unknown keys fail fast (a misspelt or unsupported switch must not be ignored)."""
        unknown = sorted(k for k in config_dict if k not in cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown reward config keys: {unknown}")
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path], config_name: Optional[str] = None) -> 'RewardConfig':
        """Create config from a YAML file: one named preset (config_name) or the whole file as a single preset.
        A relative path is resolved against this file's directory."""
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = Path(__file__).parent / yaml_path
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config file not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        if config_name is not None:
            if config_name not in yaml_content:
                raise ValueError(f"Config '{config_name}' not found in {yaml_path}. Available: {list(yaml_content.keys())}")
            return cls.from_dict(yaml_content[config_name])
        return cls.from_dict(yaml_content)


class RewardComputer:
    """Episode reward of the CircuitOp environment: 0 before the done step; at the done step the terminal metric bonus
    (constraint goals) or the improvement reward (optimization goals), 0 when the final action is invalid.

    The invalid-action penalty is applied by the trainer (apply_invalid_action_penalty in ray_trainer.py), not here.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def compute_reward(self, info: Dict[str, Any]) -> float:
        if not info.get('done', False):
            return 0.0
        goal = info.get('goal', {})
        is_final_action_valid = info.get('is_action_valid', 1) == 1
        if goal.get('mode', 'constraint') == 'constraint':
            bonus = self._compute_terminal_metric_bonus(info)
            reward = bonus if (bonus > 0 and is_final_action_valid) else 0.0
        else:
            reward = self._compute_optimization_reward(info, goal) if is_final_action_valid else 0.0
        return reward

    def _compute_optimization_reward(self, info: Dict[str, Any], goal: Dict[str, Any]) -> float:
        from circuitop.rewards import check_gain_improved, compute_performance_score
        if not check_gain_improved(info, goal, initial_gain_dB=info.get('initial_gain_dB')):
            return 0.0
        if self.config.sparse_optimization_style == 'fixed_bonus':
            return self.config.success_bonus
        metrics = {'gain_dB': info.get('gain_dB', 0.0), 'power_uW': info.get('power_uW', 1e9)}
        return compute_performance_score(metrics, dict(goal), initial_gain_dB=info.get('initial_gain_dB', 0.0),
                                         initial_power_uW=info.get('initial_power_uW', 1.0))

    def _metric_met(self, metric_name: str, info: Dict[str, Any], goal: Dict[str, Any]) -> bool:
        """Per-spec verdict used by the terminal bonus and the <spec>_pass info fields. power / bandwidth / phase_margin
        use rewards.evaluate_constraint_specs (in-band gate, PM stable_no_crossing pass, PM window, strict missing-value
        failure); gain uses the gain-tolerance check with the in-band gate on top."""
        if metric_name in ('power', 'bandwidth', 'phase_margin'):
            from circuitop.rewards import evaluate_constraint_specs  # deferred import to avoid a cycle
            metrics = {k: info.get(k) for k in (
                'gain_dB', 'differential_gain_db', 'power_uW',
                'bandwidth_MHz', 'bandwidth_kind', 'inband_flatness_db',
                'phase_margin_deg', 'phase_margin_status')}
            single_goal = dict(goal)
            single_goal['constraint_specs'] = [metric_name]
            return bool(evaluate_constraint_specs(metrics, single_goal).get(metric_name, False))

        if metric_name == 'gain':
            _ib = None if bool(goal.get('disable_validity_gates', False)) else info.get('inband_flatness_db')
            if _ib is not None and float(_ib) > 6.0:
                return False  # gain reading is unreliable under the in-band gate
            return self._gain_metric_met(
                info=info,
                goal=goal,
                tolerance_key='gain_relative_tolerance',
                default_tolerance=float(goal.get('gain_relative_tolerance', self.config.gain_relative_tolerance)),
            )

        if metric_name == 'cmrr':
            target = goal.get('cmrr_min_db')
            value = info.get('cmrr_db')
            return target is not None and value is not None and float(value) >= float(target)

        if metric_name == 'offset':
            target = goal.get('offset_abs_max_v')
            value = info.get('offset_v')
            return target is not None and value is not None and abs(float(value)) <= float(target)

        if metric_name == 'psrr_plus':
            target = goal.get('psrr_plus_min_db')
            value = info.get('psrr_plus_db')
            return target is not None and value is not None and float(value) >= float(target)

        if metric_name == 'psrr_minus':
            target = goal.get('psrr_minus_min_db')
            value = info.get('psrr_minus_db')
            return target is not None and value is not None and float(value) >= float(target)

        return False

    def _gain_metric_met(
        self,
        info: Dict[str, Any],
        goal: Dict[str, Any],
        tolerance_key: str = 'gain_relative_tolerance',
        default_tolerance: Optional[float] = None,
    ) -> bool:
        gain_target = goal.get('gain_min')
        gain_value = info.get('differential_gain_db')
        if gain_value is None:
            gain_value = info.get('gain_dB')
        if gain_target is None or gain_value is None:
            return False
        tol = goal.get(
            tolerance_key,
            default_tolerance if default_tolerance is not None else self.config.gain_relative_tolerance,
        )
        return float(gain_value) >= float(gain_target) - abs(float(gain_target)) * float(tol)

    def _compute_terminal_metric_bonus(self, info: Dict[str, Any]) -> float:
        goal = info.get('goal', {}) or {}
        gates_off = bool(goal.get('disable_validity_gates', False))
        if not gates_off:
            # gate A: no terminal metric bonus without positive gain
            gain_val = info.get('differential_gain_db')
            if gain_val is None:
                gain_val = info.get('gain_dB')
            try:
                gain_num = float(gain_val) if gain_val is not None else None
            except (TypeError, ValueError):
                gain_num = None
            if gain_num is None or gain_num <= 0.0:
                return 0.0

        bonus_map = self.config.terminal_metric_bonus
        gain_met = self._gain_metric_met(
            info=info,
            goal=goal,
            tolerance_key='terminal_bonus_gain_relative_tolerance',
            default_tolerance=self.config.terminal_bonus_gain_relative_tolerance,
        )
        _ib = None if gates_off else info.get('inband_flatness_db')
        if _ib is not None and float(_ib) > 6.0:
            gain_met = False  # gate B: the gain reading is unreliable
        if self.config.require_gain_met_for_terminal_bonus and not gain_met:
            return 0.0
        phase_margin_scored = (bool(self.config.include_phase_margin_in_terminal_bonus)
                               and goal.get('phase_margin_min_deg') is not None)

        bonus = 0.0
        if gain_met:
            bonus += float(bonus_map.get('gain', 0.0))
        if self._metric_met('power', info, goal):
            bonus += float(bonus_map.get('power', 0.0))
        if phase_margin_scored and self._metric_met('phase_margin', info, goal):
            bonus += float(bonus_map.get('phase_margin', 0.0))
        if goal.get('bandwidth_min') is not None and self._metric_met('bandwidth', info, goal):
            bonus += float(bonus_map.get('bandwidth', 0.0))
        # optional extra specs (fully differential topologies): counted when the goal sets a target
        if goal.get('offset_abs_max_v') is not None and self._metric_met('offset', info, goal):
            bonus += float(bonus_map.get('offset', 0.0))
        if goal.get('cmrr_min_db') is not None and self._metric_met('cmrr', info, goal):
            bonus += float(bonus_map.get('cmrr', 0.0))
        if goal.get('psrr_plus_min_db') is not None and self._metric_met('psrr_plus', info, goal):
            bonus += float(bonus_map.get('psrr_plus', 0.0))
        if goal.get('psrr_minus_min_db') is not None and self._metric_met('psrr_minus', info, goal):
            bonus += float(bonus_map.get('psrr_minus', 0.0))
        return bonus


def load_reward_config(config: Optional[Union[Dict[str, Any], str]] = None) -> RewardConfig:
    """Load a reward config: None (defaults), a dict, or 'path/to/file.yaml[:preset_name]'."""
    if config is None:
        loaded = RewardConfig()
    elif isinstance(config, str):
        yaml_path, _, config_name = config.partition(':')
        loaded = RewardConfig.from_yaml(yaml_path, config_name or None)
    else:
        loaded = RewardConfig.from_dict(config)
    return loaded
