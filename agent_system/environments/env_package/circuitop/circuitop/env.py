"""
CircuitOp environment: circuit-sizing episodes simulated with ngspice or Spectre.

Every observation is the dynamic state (PERFORMANCE + FEATURES) rendered by GraphXMLGenerator.
"""
import json
import os
import sys
import math
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# SpectreRunner lives in circuit/data_pre/spectre_runner.py
sys.path.insert(0, str(Path(__file__).parents[5] / "circuit" / "data_pre"))
from spectre_runner import SpectreRunner


def _tolerance_note_for_goal(goal_dict):
    """Tolerance clause appended to the goal line; values come from the preset (enriched into the goal by the env), never hard-coded."""
    try:
        g = float(goal_dict.get('gain_relative_tolerance', 0.02)) * 100
        p = float(goal_dict.get('power_relative_tolerance', 0.05)) * 100
        b = float(goal_dict.get('bandwidth_relative_tolerance', 0.05)) * 100
        m = float(goal_dict.get('phase_margin_relative_tolerance', 0.05)) * 100
    except (TypeError, ValueError):
        return ""
    def _f(x):
        return f"{x:g}%"
    if p == b == m:
        if g == p:
            return f" (each spec has {_f(g)} tolerance)"
        return f" (tolerance: gain {_f(g)}, others {_f(p)})"
    return f" (tolerance: gain {_f(g)}, power {_f(p)}, bandwidth {_f(b)}, PM {_f(m)})"


from .rewards import (
    compute_performance_score,
    evaluate_constraint_specs,
    has_positive_gain,
    check_core_success,
    check_goal_met,
)
from .state_utils import (
    get_admissible_actions,
    build_simulation_bundle_from_dataset,
    construct_netlist_from_dataset,
    construct_observation_netlist,
    find_geometry_bound_violations,
    find_near_bound_params,
    format_tech_geometry_bounds,
    get_netlist_for_observation,
    _extract_parameter_map,
)
from .action_tool import NetlistActionTool
from .observation_builder import CircuitObservationBuilder
from .action_validator import ActionValidator

# Import reward configuration (one level up from circuitop/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from reward_config import RewardConfig, RewardComputer
from pathlib import Path as _P
_REPO_ROOT = str(_P(__file__).resolve().parents[4])



class CircuitOpEnv:
    """Gym-like environment for circuit optimization; returns undiscounted per-step rewards."""
    
    def __init__(self,
                 workdir: str,
                 max_steps: int = 16,
                 circuit_name: str = "default_circuit",
                 goal: Optional[Dict[str, float]] = None,
                 reward_config: Optional[RewardConfig] = None):
        """
        Args:
            workdir: Working directory for simulations
            max_steps: Maximum steps per episode (hard limit)
            circuit_name: Name of circuit being optimized
            goal: target dict (gain_min, power_max, bandwidth_min, phase_margin_min_deg, phase_margin_max_deg)
            reward_config: RewardConfig of the preset
        """
        self.workdir = Path(workdir)
        self.max_steps = int(max_steps)
        self.circuit_name = circuit_name
        
        # Goal specifications (used for termination and info)
        self.goal = goal or {'gain_min': 20.0, 'power_max': 100.0}
        
        # Initialize reward computer with config BEFORE enriching goal
        self.reward_config = reward_config or RewardConfig()
        self.reward_computer = RewardComputer(self.reward_config)
        
        # Enrich goal with the reward config's parameters (dataset goals carry only the targets)
        enriched_fields = {
            'constraint_specs': list(self.reward_config.constraint_specs),
            'allow_further_exploration_after_goal': self.reward_config.allow_further_exploration_after_goal,
            'gain_improvement_tolerance_dB': self.reward_config.gain_improvement_tolerance_dB,
            'gain_relative_tolerance': self.reward_config.gain_relative_tolerance,
            'terminal_bonus_gain_relative_tolerance': self.reward_config.terminal_bonus_gain_relative_tolerance,
            'power_relative_tolerance': self.reward_config.power_relative_tolerance,
            'bandwidth_relative_tolerance': self.reward_config.bandwidth_relative_tolerance,
            'phase_margin_relative_tolerance': self.reward_config.phase_margin_relative_tolerance,
            'action_change_mode': self.reward_config.action_change_mode,
            'goal_prompt_style': self.reward_config.goal_prompt_style,
        }
        if self.reward_config.goal_optimization_hint:
            enriched_fields['optimization_hint'] = self.reward_config.goal_optimization_hint
        # A dataset row may set allow_further_exploration_after_goal itself; the config value is only a default for rows
        # that do not.
        if 'allow_further_exploration_after_goal' in self.goal:
            enriched_fields['allow_further_exploration_after_goal'] = bool(self.goal['allow_further_exploration_after_goal'])
        # The flag rides on the goal so every spec/success/bonus check that reads self.goal sees it
        if bool(getattr(self.reward_config, 'disable_validity_gates', False)):
            enriched_fields['disable_validity_gates'] = True
        self.goal = {**self.goal, **enriched_fields}
        # Prompt style read by the manager's templates; a dataset-provided value wins
        self.goal.setdefault('prompt_goal_style', self.reward_config.goal_prompt_style)

        # Create workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        
        # Paths to circuit files (assume bootstrap done externally)
        self.cache_dir = self.workdir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.netlist_scs_path = self.cache_dir / "netlist.scs"
        
        # CIRCUIT_SIM_BACKEND: ngspice (default) or spectre; unknown values fail fast.
        _backend = os.environ.get("CIRCUIT_SIM_BACKEND", "ngspice").strip().lower()
        if _backend == "ngspice":
            from ngspice_runner import NgspiceRunner as _RunnerCls
            scratch_dir = self.workdir / "ngspice_scratch"
        elif _backend == "spectre":
            _RunnerCls = SpectreRunner
            scratch_dir = self.workdir / "spectre_scratch"
        else:
            raise RuntimeError(f"Unknown CIRCUIT_SIM_BACKEND='{_backend}' (spectre|ngspice)")
        self.simulator = _RunnerCls(
            scratch_dir=str(scratch_dir),
        )
        
        # Initialize action tool, observation builder, and validator
        self.obs_builder = CircuitObservationBuilder(circuit_name=circuit_name)
        self.action_validator = ActionValidator()
        self.action_tool = NetlistActionTool(handle_to_id={})
        
        # Episode state (initialized in reset)
        self.t = 0
        self.netlist_text = ""
        self.main_instance_name = None
        self.support_instance_names = []
        self.prev_perf_score = 0.0
        self.initial_obs_xml = None  # For loading static structure from dataset
        self.task_data = None  # Dataset task data with admissible_actions field
        self.representation = 'xml'  # Default to xml format (handles)
        
        # Track initial metrics for sparse no-goal mode
        self.initial_gain_dB = None
        self.initial_power_uW = None
        
        # Cache last valid observation for invalid action feedback
        self.last_valid_obs = None
        # Cache most recent observation shown to the model, regardless of validity.
        self.last_obs = None
        
        # Whether the goal was met at the previous step (reported as already_achieved_goal)
        self.goal_met_in_prev_step = False
        
        # Best-achieved metrics tracking: snapshot from the single step with the
        # highest performance score.
        self.best_achieved_metrics: Dict[str, Any] = {}
        self.best_turn_score: float = -float('inf')


    def _cache_observation(self, obs: Optional[str], *, valid: bool = False) -> Optional[str]:
        self.last_obs = obs
        if valid:
            self.last_valid_obs = obs
        return obs

    def _get_last_observation(self) -> str:
        if self.last_obs:
            return self.last_obs
        if self.last_valid_obs:
            return self.last_valid_obs
        return ""

    def _goal_constraint_specs(self) -> Optional[list]:
        specs = self.goal.get(
            'constraint_specs',
            getattr(self.reward_config, 'constraint_specs', None),
        )
        if specs is None:
            return None
        if isinstance(specs, str):
            return [specs]
        try:
            return [str(s) for s in specs]
        except TypeError:
            return None

    def _goal_allows_further_exploration_after_goal(self) -> bool:
        return bool(
            self.goal.get(
                'allow_further_exploration_after_goal',
                getattr(self.reward_config, 'allow_further_exploration_after_goal', False)
            )
        )

    def _validate_constraint_specs_config(self) -> None:
        if self.goal.get('mode', 'constraint') != 'constraint':
            return
        gate_specs = self._goal_constraint_specs()
        if not gate_specs:
            raise RuntimeError("Invalid reward config: 'constraint_specs' must be set.")
    # ── episode-state snapshot ──
    # SLIC continuation sync copies the selected branch's state (deep copy of all attributes) to its
    # group peers instead of replaying the history.
    def get_state(self) -> dict:
        import copy
        return copy.deepcopy(self.__dict__)

    def set_state(self, state: dict) -> None:
        import copy
        self.__dict__.update(copy.deepcopy(state))

    def _resolve_termination_goal_met(self, metrics: Dict[str, Any], goal_met: bool) -> bool:
        if self.goal.get('mode', 'constraint') == 'constraint':
            # the episode stops when every spec in constraint_specs passes
            gate_specs = self._goal_constraint_specs()
            if not gate_specs:
                raise RuntimeError(
                    "constraint_specs is missing during termination check."
                )
            gate_goal = dict(self.goal)
            gate_goal['constraint_specs'] = gate_specs
            gate_results = evaluate_constraint_specs(metrics, gate_goal)
            return all(bool(gate_results.get(spec, False)) for spec in gate_specs)
        return bool(goal_met)
        
    # =========================================================================
    # PUBLIC API - Required by environment wrapper
    # =========================================================================
    
    def reset(self, initial_obs_xml: Optional[str] = None, task_data: Optional[Dict] = None, representation: str = 'xml',
              topology: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            initial_obs_xml: Optional initial observation XML from dataset
            task_data: Dataset task dict with 'admissible_actions' and 'structure' fields
            representation: 'xml' (uses @handles) or 'netlist' (uses device_ids)
            topology: the row's topology; replaces circuit_name for this episode. Default:
                task_data.metadata.topology_name, else circuit_name.
        
        Returns:
            (observation, info) tuple
        """
        # Reset episode counters
        self.t = 0
        self.representation = representation
        if task_data is not None:
            # the row names its topology once; the simulator reads it as task_data.metadata.topology_name
            named = (task_data.get('metadata') or {}).get('topology_name')
            if topology and named and named != topology:
                raise ValueError(f"row topology {topology!r} differs from task_data.metadata.topology_name={named!r}")
            self.circuit_name = topology or named or self.circuit_name
            task_data = {**task_data, 'metadata': {**(task_data.get('metadata') or {}), 'topology_name': self.circuit_name}}
        self.task_data = task_data
        self._validate_constraint_specs_config()
        
        
        # Construct netlist from task_data (dataset structure + params)
        if task_data:
            simulation_bundle = build_simulation_bundle_from_dataset(task_data)
            self.main_instance_name = simulation_bundle.get('main_instance_name')
            self.support_instance_names = simulation_bundle.get('support_instance_names', []) or []
            # Runnable netlist: dataset core netlist + initial params inside the simulator header/footer
            self.netlist_text = construct_netlist_from_dataset(task_data, representation)
            # Write initial netlist for simulation
            self.netlist_scs_path.write_text(self.netlist_text, encoding='utf-8')
        else:
            raise ValueError("No netlist found in task_data['structure']['netlist']")
        
        # Store initial obs for static structure extraction
        self.initial_obs_xml = initial_obs_xml
        
        # Initialize observation builder with static structure
        # Use structure XML (with GRAPH_NODE_IDMAP) from task_data, not observation XML
        structure_xml = task_data.get('structure', {}).get('xml', '') if task_data else None
        # Use circuit-only netlist for observation so testbench is not shown
        netlist_for_obs = construct_observation_netlist(task_data, self.netlist_text)
        if structure_xml:
            # Extract static structure from dataset XML (contains IDMAP for handle mappings)
            self.obs_builder.reset(
                netlist_for_obs,
                static_xml=structure_xml,
                representation=representation,
                main_instance_name=self.main_instance_name,
                support_instance_names=self.support_instance_names,
            )
        else:
            # Generate from netlist
            self.obs_builder.reset(
                netlist_for_obs,
                representation=representation,
                main_instance_name=self.main_instance_name,
                support_instance_names=self.support_instance_names,
            )
        
        # Load action validator from dataset
        if task_data and 'admissible_actions' in task_data:
            structure_xml = task_data.get('structure', {}).get('xml', '')
            if structure_xml:
                self.action_validator.load_from_dataset(
                    admissible_actions_dict=task_data['admissible_actions'],
                    structure_xml=task_data['structure']['xml'],
                    representation=representation
                )
                # Optional override for action percentage range constraints.
                self.action_validator.set_percentage_constraints(
                    enforce=getattr(self.reward_config, 'enforce_action_percentage_range', True),
                    pct_min=getattr(self.reward_config, 'action_percentage_min', None),
                    pct_max=getattr(self.reward_config, 'action_percentage_max', None),
                )
                self.action_validator.set_action_change_mode(
                    getattr(self.reward_config, 'action_change_mode', 'percentage')
                )
                self.action_validator.set_prompt_target_limit(
                    getattr(self.reward_config, 'action_space_prompt_max_targets', 20)
                )
                # Update action tool with handle mapping from validator
                self.action_tool.handle_to_id = self.action_validator.handle_to_id
        
        # Get initial metrics from task_data (load once for both internal state and info dict)
        initial_metrics = {}
        initial_op_data = {}
        if task_data and 'initial_state' in task_data:
            initial_state_metrics = task_data['initial_state'].get('metrics', {})
            initial_op_data = initial_state_metrics.get('op_data') or {}
            initial_gain = initial_state_metrics.get('gain_dB')
            self.initial_gain_dB = 0.0 if initial_gain is None else float(initial_gain)

            initial_power = initial_state_metrics.get('power_uW')
            # a stored start without a power reading stays NaN, as in the reset-refresh path below (0.0 would pass the power
            # spec of the initial performance score when the refresh fails).
            self.initial_power_uW = float('nan') if initial_power is None else float(initial_power)

            initial_metrics['gain_dB'] = self.initial_gain_dB
            initial_metrics['differential_gain_db'] = initial_state_metrics.get('differential_gain_db')
            initial_metrics['power_uW'] = self.initial_power_uW
            initial_metrics['bandwidth_MHz'] = 0.0 if initial_state_metrics.get('bandwidth_MHz') is None else float(initial_state_metrics.get('bandwidth_MHz'))
            initial_metrics['phase_margin_deg'] = initial_state_metrics.get('phase_margin_deg')
            initial_metrics['offset_v'] = initial_state_metrics.get('offset_V', initial_state_metrics.get('offset_v'))
            initial_metrics['cmrr_db'] = initial_state_metrics.get('cmrr_db')
            initial_metrics['psrr_plus_db'] = initial_state_metrics.get('psrr_plus_db')
            initial_metrics['psrr_minus_db'] = initial_state_metrics.get('psrr_minus_db')
        else:
            # Fallback: run initial simulation to get baseline
            bounds_error = self._get_geometry_bounds_error(self.netlist_text)
            if bounds_error:
                initial_sim = {
                    'gain': 0.0,
                    'power': 0.0,
                    'bandwidth': 0.0,
                    'phase_margin_deg': None,
                    'differential_gain_db': None,
                    'offset_v': None,
                    'cmrr_db': None,
                    'psrr_plus_db': None,
                    'psrr_minus_db': None,
                    'op_data': {},
                }
            else:
                initial_sim = self.simulator.run_simulation(self.netlist_text, timeout=900, threads=1)
            self.initial_gain_dB = initial_sim.get('gain', 0.0)
            initial_power = initial_sim.get('power')
            self.initial_power_uW = initial_power * 1e6 if initial_power is not None else float('nan')
            initial_metrics['gain_dB'] = self.initial_gain_dB
            initial_metrics['power_uW'] = self.initial_power_uW
            # the runner emits bandwidth_MHz directly (MHz everywhere); None keeps its "not measurable" meaning
            initial_metrics['bandwidth_MHz'] = initial_sim.get('bandwidth_MHz')
            initial_metrics['bandwidth_kind'] = initial_sim.get('bandwidth_kind')
            initial_metrics['inband_flatness_db'] = initial_sim.get('inband_flatness_db')
            initial_metrics['phase_margin_status'] = initial_sim.get('phase_margin_status')
            initial_metrics['phase_margin_deg'] = initial_sim.get('phase_margin_deg')
            initial_metrics['differential_gain_db'] = initial_sim.get('differential_gain_db')
            initial_metrics['offset_v'] = initial_sim.get('offset_v')
            initial_metrics['cmrr_db'] = initial_sim.get('cmrr_db')
            initial_metrics['psrr_plus_db'] = initial_sim.get('psrr_plus_db')
            initial_metrics['psrr_minus_db'] = initial_sim.get('psrr_minus_db')
            initial_op_data = initial_sim.get('op_data', {}) or {}

        if task_data and 'initial_state' in task_data:
            # Initial metrics come from a run-time re-simulation (matches the observation text); the stored
            # dataset values are the fallback when it returns no operating point.
            refresh = self._simulate_initial_operating_point(max_attempts=3)
            if refresh and refresh.get('op_data'):
                initial_op_data = refresh.get('op_data', {}) or initial_op_data
                _rg, _rp = refresh.get('gain'), refresh.get('power')
                self.initial_gain_dB = float(_rg) if _rg is not None else float('nan')
                self.initial_power_uW = float(_rp) * 1e6 if _rp is not None else float('nan')
                initial_metrics.update({
                    'gain_dB': self.initial_gain_dB,
                    'power_uW': self.initial_power_uW,
                    'bandwidth_MHz': refresh.get('bandwidth_MHz'),
                    'bandwidth_kind': refresh.get('bandwidth_kind'),
                    'inband_flatness_db': refresh.get('inband_flatness_db'),
                    'phase_margin_status': refresh.get('phase_margin_status'),
                    'phase_margin_deg': refresh.get('phase_margin_deg'),
                    'differential_gain_db': refresh.get('differential_gain_db'),
                    'offset_v': refresh.get('offset_v'),
                    'cmrr_db': refresh.get('cmrr_db'),
                    'psrr_plus_db': refresh.get('psrr_plus_db'),
                    'psrr_minus_db': refresh.get('psrr_minus_db'),
                })
            elif refresh and refresh.get('error_type') == 'spectre_crash':
                raise RuntimeError(
                    f"Initial metrics refresh crashed in Spectre: {refresh.get('feedback')}")

        # Initial performance score; the first step's delta is measured against it
        self.initial_perf_score = float(self._compute_performance_score(initial_metrics))
        self.prev_perf_score = self.initial_perf_score
        self.goal_met_in_prev_step = False
        
        # Track current metrics for error handling (use last valid state, not initial)
        self.current_metrics = {
            'gain_dB': initial_metrics['gain_dB'],
            'differential_gain_db': initial_metrics.get('differential_gain_db'),
            'power_uW': initial_metrics['power_uW'],
            'bandwidth_MHz': initial_metrics.get('bandwidth_MHz'),  # None = not measurable; never substitute 0
            'bandwidth_kind': initial_metrics.get('bandwidth_kind'),
            'inband_flatness_db': initial_metrics.get('inband_flatness_db'),
            'phase_margin_status': initial_metrics.get('phase_margin_status'),
            'phase_margin_deg': initial_metrics.get('phase_margin_deg'),
            'offset_v': initial_metrics.get('offset_v'),
            'cmrr_db': initial_metrics.get('cmrr_db'),
            'psrr_plus_db': initial_metrics.get('psrr_plus_db'),
            'psrr_minus_db': initial_metrics.get('psrr_minus_db'),
        }
        # Best snapshot is recorded only from gain>0 states.
        self.best_achieved_metrics = {}
        self.best_turn_score = -float('inf')

        initial_bounds_error = self._get_geometry_bounds_error(self.netlist_text)
        if initial_bounds_error:
            precheck_error_type = 'design_error'
            obs = self._build_design_error_observation(
                error_msg=initial_bounds_error,
                metrics=self.current_metrics,
                step=0,
                fallback_obs=initial_obs_xml,
                netlist_text=self.netlist_text,
                status=f"INVALID - {precheck_error_type}",
            )
            self._cache_observation(obs, valid=True)
            info = self._build_step_info(
                metrics=initial_metrics,
                extra={
                    'error': initial_bounds_error,
                    'error_type': precheck_error_type,
                },
            )
            return obs, info
        
        # Initial observation: rebuild when the dataset XML is stale/incomplete so the
        # prompt always has current operating points and enriched parameter tokens.
        obs = self._build_initial_observation(
            initial_metrics=initial_metrics,
            initial_obs_xml=initial_obs_xml,
            initial_op_data=initial_op_data,
        )
        # The measurement conventions are carried by the optimization hint: a run must keep the
        # goal_with_optimization_hint prompt style, otherwise the conventions appear nowhere in the prompt.

        # Cache initial observation for invalid action handling
        self._cache_observation(obs, valid=True)
        
        # Info dict (include goal for env_manager to extract)
        info = self._build_step_info(metrics=initial_metrics)
        
        return obs, info

    def _maybe_update_best_achieved(
        self,
        metrics: Dict[str, Any],
        perf_score_post: Optional[float] = None,
    ) -> None:
        """Snapshot metrics if this step's perf score beats current best."""
        score = (
            float(perf_score_post)
            if perf_score_post is not None
            else self._compute_performance_score(metrics)
        )
        if score > self.best_turn_score:
            self.best_turn_score = score
            self.best_achieved_metrics = dict(metrics)

    def _compute_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Compute performance score with env/reward-config options merged in."""
        score_goal = dict(self.goal or {})
        score_goal['disable_validity_gates'] = bool(getattr(self.reward_config, 'disable_validity_gates', False))
        score_goal['success_margin_weight'] = float(getattr(self.reward_config, 'success_margin_weight', 0.0))
        score_goal['success_slack_cap'] = float(getattr(self.reward_config, 'success_slack_cap', 1.0))
        return float(
            compute_performance_score(
                metrics,
                score_goal,
                initial_gain_dB=self.initial_gain_dB,
                initial_power_uW=self.initial_power_uW,
            )
        )

    def _compute_best_turn_state_score(self, metrics: Dict[str, Any]) -> float:
        """Best-turn state score used only for selecting best_achieved_metrics.

        If gain is non-positive, only gain spec is scored to avoid selecting
        deceptive states with low power / high bandwidth but invalid gain.
        """
        gain_val = metrics.get('gain_dB')
        if gain_val is None:
            gain_val = metrics.get('differential_gain_db')
        gain_val = 0.0 if gain_val is None else float(gain_val)

        if gain_val > 0.0:
            return self._compute_performance_score(metrics)

        gain_only_goal = dict(self.goal or {})
        gain_only_goal['constraint_specs'] = ['gain']
        return float(
            compute_performance_score(
                metrics,
                gain_only_goal,
                initial_gain_dB=self.initial_gain_dB,
                initial_power_uW=self.initial_power_uW,
            )
        )

    def _is_complex_topology(self) -> bool:
        if self.task_data:
            metadata = self.task_data.get('metadata', {}) or {}
            if metadata.get('complexity') == 'complex':
                return True
        current_metrics = getattr(self, 'current_metrics', {}) or {}
        return any(
            current_metrics.get(k) is not None
            for k in ('differential_gain_db', 'offset_v', 'cmrr_db', 'psrr_plus_db', 'psrr_minus_db')
        )

    def _extract_rollout_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metrics = metrics or getattr(self, 'current_metrics', {}) or {}
        bandwidth_mhz = metrics.get('bandwidth_MHz')   # MHz is the only bandwidth key; no Hz fallback
        power_uw = metrics.get('power_uW')
        if power_uw is None and metrics.get('power') is not None:
            power_uw = float(metrics['power']) * 1e6
        return {
            'gain_dB': metrics.get('gain_dB', metrics.get('gain')),
            'power_uW': power_uw,
            'bandwidth_MHz': bandwidth_mhz,
            'bandwidth_kind': metrics.get('bandwidth_kind'),          # 'f3db' | None
            'inband_flatness_db': metrics.get('inband_flatness_db'),  # raw value behind the in-band flatness gate
            'phase_margin_deg': metrics.get('phase_margin_deg'),
            'phase_margin_status': metrics.get('phase_margin_status'),  # measured | stable_no_crossing | undefined
            'differential_gain_db': metrics.get('differential_gain_db'),
            'offset_v': metrics.get('offset_v', metrics.get('offset_V')),
            'cmrr_db': metrics.get('cmrr_db'),
            'psrr_plus_db': metrics.get('psrr_plus_db'),
            'psrr_minus_db': metrics.get('psrr_minus_db'),
        }

    def _build_step_info(
        self,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        done: bool = False,
        is_action_valid: int = 1,
        perf_score_post: Optional[float] = None,
        perf_score_pre: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Info dict for every step/reset path: the common keys plus path-specific ``extra``."""
        rollout_metrics = self._extract_rollout_metrics(metrics)
        core_success = check_core_success(rollout_metrics, self.goal)
        info: Dict[str, Any] = {
            'done': bool(done),
            'success': 1 if core_success else 0,
            'core_success': 1 if core_success else 0,
            'is_action_valid': is_action_valid,
            'admissible_commands': get_admissible_actions(self.action_validator),
            'goal': self.goal,
            'initial_gain_dB': self.initial_gain_dB,
            'initial_power_uW': self.initial_power_uW,
            'data_source': self.circuit_name,
            'already_achieved_goal': getattr(self, 'goal_met_in_prev_step', False),
            'perf_score_post': float(perf_score_post if perf_score_post is not None
                                     else getattr(self, 'prev_perf_score', 0.0)),
            'perf_score_pre': float(perf_score_pre if perf_score_pre is not None
                                    else getattr(self, 'prev_perf_score', 0.0)),
            'perf_score_initial': float(getattr(self, 'initial_perf_score', 0.0)),
            'best_achieved_metrics': dict(getattr(self, 'best_achieved_metrics', {})),
            **rollout_metrics,
        }
        if extra:
            info.update(extra)

        goal = info.get('goal', {}) or {}
        metric_target_keys = {
            'gain': goal.get('gain_min'),
            'power': goal.get('power_max'),
            'bandwidth': goal.get('bandwidth_min'),
            'phase_margin': goal.get('phase_margin_min_deg'),
            'offset': goal.get('offset_abs_max_v'),
            'cmrr': goal.get('cmrr_min_db'),
            'psrr_plus': goal.get('psrr_plus_min_db'),
            'psrr_minus': goal.get('psrr_minus_min_db'),
        }
        for metric_name, target_value in metric_target_keys.items():
            if target_value is None:
                continue
            metric_pass = self.reward_computer._metric_met(metric_name, info, goal)
            if metric_name != 'gain' and not has_positive_gain(info) and not bool(goal.get('disable_validity_gates', False)):
                metric_pass = False
            info[f'{metric_name}_pass'] = 1.0 if metric_pass else 0.0

        # Resulting design params after this step (testbench params such as cload or vcm excluded)
        try:
            raw_params = _extract_parameter_map(self.netlist_text)
            initial_params = (self.task_data or {}).get('initial_state', {}).get('params', {})
            if initial_params:
                info['resulting_params'] = {k: v for k, v in raw_params.items() if k in initial_params}
            else:
                info['resulting_params'] = raw_params
        except Exception:
            info['resulting_params'] = {}
        info['resulting_metrics'] = dict(getattr(self, 'current_metrics', {}) or {})

        return info

    def _compute_step_reward(self, info: Dict[str, Any]) -> float:
        """Step reward; first adds ``metrics_pre`` and ``best_achieved_metrics`` to *info*."""
        info['metrics_pre'] = getattr(self, '_step_metrics_pre', {})
        info['best_achieved_metrics'] = dict(getattr(self, 'best_achieved_metrics', {}))
        return self.reward_computer.compute_reward(info)

    def _get_obs_constraint_specs(self) -> Optional[set]:
        """Return the active constraint_specs as a set, or None if all specs should be shown."""
        goal = getattr(self, 'goal', None)
        if not goal:
            return None
        specs = goal.get('constraint_specs')
        if specs is not None:
            return set(specs)
        return None

    def _build_performance_block(self, status: str, error: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> List[str]:
        if metrics is None:
            metrics = getattr(self, 'current_metrics', {}) or {}
        if not metrics:
            # Explicitly empty dict: omit all metric values, show only status/error
            obs_lines = ["<PERFORMANCE>"]
            obs_lines.append(f"<Status>{status}</Status>")
            if error:
                obs_lines.append(f"<Error>{error}</Error>")
            obs_lines.append("</PERFORMANCE>")
            return obs_lines
        obs_lines = ["<PERFORMANCE>"]
        obs_lines.append(f"<Status>{status}</Status>")
        if error:
            obs_lines.append(f"<Error>{error}</Error>")

        cs = self._get_obs_constraint_specs()

        gain_val = metrics.get('gain_dB', metrics.get('gain', 0.0)) or 0.0
        power_val = metrics.get('power_uW')
        bw_val = metrics.get('bandwidth_MHz', 0.0)
        pm_val = metrics.get('phase_margin_deg')

        if cs is None or 'gain' in cs:
            # the label carries the measurement convention (gain sampled at 1 kHz)
            obs_lines.append(f"<GainAt1kHz>{self._format_gain_db(gain_val) or '0.0 dB'}</GainAt1kHz>")
        if power_val is not None and (cs is None or 'power' in cs):
            obs_lines.append(f"<Power>{self._format_power_uw(power_val)}</Power>")

        try:
            bw_num = float(bw_val)
        except (TypeError, ValueError):
            bw_num = 0.0
        if bw_num > 0 and abs(bw_num - 9999.99) > 1e-2 and (cs is None or 'bandwidth' in cs):
            obs_lines.append(f"<Bandwidth3dB>{self._format_bandwidth_human_mhz(bw_num)}</Bandwidth3dB>")
        elif (metrics.get('bandwidth_MHz') is None and 'bandwidth_kind' in metrics
              and (cs is None or 'bandwidth' in cs)):
            obs_lines.append("<Bandwidth3dB>not measurable</Bandwidth3dB>")

        if pm_val is not None and (cs is None or 'phase_margin' in cs):
            obs_lines.append(f"<PhaseMargin>{self._format_integer_unit(pm_val, 'deg')}</PhaseMargin>")
        elif metrics.get('phase_margin_status') == 'stable_no_crossing' and (cs is None or 'phase_margin' in cs):
            # no 0 dB crossing but a clean in-band response -> spec passes; show it explicitly instead of leaving a blank
            obs_lines.append("<PhaseMargin>stable (no 0 dB crossing; spec passed)</PhaseMargin>")
        _ibm = metrics.get('inband_flatness_db')
        _gates_off = bool((self.goal or {}).get('disable_validity_gates', False))
        if _ibm is not None:
            obs_lines.append(f"<InBandFlatness>{float(_ibm):.1f} dB{'' if _gates_off else ' (limit 6)'}</InBandFlatness>")
        # the causal Validity line is part of PERFORMANCE; with disable_validity_gates the gates do not exist, so no line
        if _gates_off:
            pass
        elif not has_positive_gain(metrics):
            obs_lines.append("<Validity>non-positive gain; power/bandwidth/PM not meaningful until gain > 0</Validity>")
        elif _ibm is not None and float(_ibm) > 6.0:
            obs_lines.append(f"<Validity>in-band peaking {float(_ibm):.1f} dB exceeds 6 dB; gain/bandwidth/PM readings unreliable</Validity>")

        if self._is_complex_topology():
            diff_gain = metrics.get('differential_gain_db')
            offset_v = metrics.get('offset_v')
            cmrr_db = metrics.get('cmrr_db')
            psrr_plus_db = metrics.get('psrr_plus_db')
            psrr_minus_db = metrics.get('psrr_minus_db')
            if diff_gain is not None and cs is None:
                obs_lines.append(f"<DifferentialGain>{self._format_gain_db(diff_gain)}</DifferentialGain>")
            if offset_v is not None and (cs is None or 'offset' in cs):
                obs_lines.append(f"<Offset>{self._format_scaled_value(offset_v, [(1.0, 'V'), (1e-3, 'mV'), (1e-6, 'uV'), (1e-9, 'nV')])}</Offset>")
            if cmrr_db is not None and (cs is None or 'cmrr' in cs):
                obs_lines.append(f"<CMRR>{self._format_gain_db(cmrr_db)}</CMRR>")
            if psrr_plus_db is not None and (cs is None or 'psrr_plus' in cs):
                obs_lines.append(f"<PSRRPlus>{self._format_gain_db(psrr_plus_db)}</PSRRPlus>")
            if psrr_minus_db is not None and (cs is None or 'psrr_minus' in cs):
                obs_lines.append(f"<PSRRMinus>{self._format_gain_db(psrr_minus_db)}</PSRRMinus>")

        # Opt-in (CIRCUITOP_BOUNDS_ALERT=1): flag params one typical action away from a geometry bound
        if os.environ.get("CIRCUITOP_BOUNDS_ALERT") == "1" and self.netlist_text:
            try:
                near = find_near_bound_params(self.netlist_text, topology=self._topology_name())
                parts = []
                if near["near_floor"]:
                    parts.append("near lower bound (a large decrease would be invalid): " + ", ".join(near["near_floor"][:8]))
                if near["near_cap"]:
                    parts.append("near upper bound (a large increase would be invalid): " + ", ".join(near["near_cap"][:8]))
                if parts:
                    obs_lines.append(f"<BoundsAlert>{'; '.join(parts)}</BoundsAlert>")
            except Exception as e:
                raise RuntimeError(f"BOUNDS_ALERT failed (fail fast): {e}")
        obs_lines.append("</PERFORMANCE>")
        return obs_lines

    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    @classmethod
    def _format_gain_db(cls, value: Any) -> Optional[str]:
        if not cls._is_finite_number(value):
            return None
        return f"{float(value):.1f} dB"

    @classmethod
    def _format_power_uw(cls, value_uw: Any) -> str:
        if not cls._is_finite_number(value_uw):
            return "N/A"
        return f"{int(round(float(value_uw)))} uW"

    @classmethod
    def _format_integer_unit(cls, value: Any, unit: str) -> str:
        if not cls._is_finite_number(value):
            return "N/A"
        return f"{int(round(float(value)))} {unit}"

    @classmethod
    def _format_scaled_value(cls, value: Any, units: List[Tuple[float, str]]) -> str:
        if not cls._is_finite_number(value):
            return "N/A"
        abs_value = abs(float(value))
        chosen_scale, chosen_unit = units[-1]
        for scale, unit in units:
            if abs_value >= scale:
                chosen_scale, chosen_unit = scale, unit
                break
        return f"{int(round(float(value) / chosen_scale))} {chosen_unit}"

    @classmethod
    def _format_bw_mhz_3sig(cls, value_mhz: Any) -> str:
        """Adaptive unit (GHz/MHz/kHz/Hz) at 3 significant digits, for goal/spec thresholds."""
        if not cls._is_finite_number(value_mhz):
            return "N/A"
        hz = float(value_mhz) * 1e6
        for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
            if abs(hz) >= scale:
                return f"{hz / scale:.3g} {unit}"
        return f"{hz:.3g} Hz"

    @classmethod
    def _format_bandwidth_human_mhz(cls, value_mhz: Any) -> str:
        if not cls._is_finite_number(value_mhz):
            return "N/A"
        value_hz = float(value_mhz) * 1e6
        return cls._format_scaled_value(
            value_hz,
            [(1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")],
        )

    @staticmethod
    def _observation_has_operating_points(obs_text: Optional[str]) -> bool:
        if not obs_text:
            return False
        return any(
            token in obs_text
            for token in ("Vgs=", "Vds=", "Vth=", "Vdsat=", "Ids=", "Gm=", "region=", "Vdrop=", " I=")
        )

    @staticmethod
    def _extract_features_section(obs_text: Optional[str]) -> Optional[str]:
        if not obs_text:
            return None
        import re
        features_match = re.search(r'(<FEATURES>.*?</FEATURES>)', obs_text, re.DOTALL)
        if not features_match:
            return None
        return features_match.group(1)

    def _normalize_op_data_for_observation(self, op_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not op_data:
            return {}

        normalized: Dict[str, Any] = {}
        path_priority = ("OP", "ADM", "DC", "ACM", "PSRRP", "PSRRN")
        main_instance_name = self.main_instance_name or "X0"
        support_instance_names = sorted(set(self.support_instance_names or []), key=len, reverse=True)

        for path_name in path_priority:
            main_prefix = f"XDUT_{path_name}."
            main_named_prefix = f"{main_instance_name}_{path_name}."

            for inst_name, values in op_data.items():
                if inst_name.startswith(main_prefix):
                    mapped = f"{main_instance_name}.{inst_name[len(main_prefix):]}"
                    normalized.setdefault(mapped, values)
                    continue

                if inst_name.startswith(main_named_prefix):
                    mapped = f"{main_instance_name}.{inst_name[len(main_named_prefix):]}"
                    normalized.setdefault(mapped, values)
                    continue

                for support_name in support_instance_names:
                    support_re = rf"^{re.escape(support_name)}_{path_name}_\d+\.(.+)$"
                    match = re.match(support_re, inst_name)
                    if match:
                        mapped = f"{support_name}.{match.group(1)}"
                        normalized.setdefault(mapped, values)
                        break

        return normalized or dict(op_data)

    def _simulate_initial_operating_point(self, max_attempts: int = 3) -> Optional[Dict[str, Any]]:
        last_result = None
        for attempt in range(1, max_attempts + 1):
            sim_result = self.simulator.run_simulation(self.netlist_text, timeout=900, threads=1)
            last_result = sim_result
            if sim_result.get('op_data'):
                return sim_result
        return last_result

    def _topology_name(self) -> Optional[str]:
        if self.task_data:
            return (self.task_data.get('metadata', {}) or {}).get('topology_name')
        return None

    def _get_geometry_bounds_error(self, netlist_text: str) -> Optional[str]:
        topology = self._topology_name()
        violations = find_geometry_bound_violations(netlist_text, topology=topology)
        if not violations:
            return None
        preview = "; ".join(violations[:5])
        if len(violations) > 5:
            preview += f"; ... ({len(violations) - 5} more)"
        return (
            "Design error: transistor width/length out of allowed range. "
            f"Violations: {preview}. "
            f"{format_tech_geometry_bounds(topology=topology)}"
        )

    def _metrics_for_observation_builder(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'gain': float(metrics.get('gain_dB', metrics.get('gain', 0.0)) or 0.0),
            'power': float(metrics.get('power_uW', 0.0) or 0.0) / 1e6,
            'bandwidth_MHz': float(metrics.get('bandwidth_MHz', 0.0) or 0.0),
            'phase_margin_deg': metrics.get('phase_margin_deg'),
            'differential_gain_db': metrics.get('differential_gain_db'),
            'offset_v': metrics.get('offset_v'),
            'cmrr_db': metrics.get('cmrr_db'),
            'psrr_plus_db': metrics.get('psrr_plus_db'),
            'psrr_minus_db': metrics.get('psrr_minus_db'),
        }

    def _build_design_error_observation(
        self,
        error_msg: str,
        metrics: Dict[str, Any],
        step: int,
        fallback_obs: Optional[str] = None,
        netlist_text: Optional[str] = None,
        status: str = "INVALID - design_error",
        omit_perf_metrics: bool = False,
    ) -> str:
        obs_lines = self._build_performance_block(
            status=status,
            error=error_msg,
            metrics={} if omit_perf_metrics else metrics,
        )
        features_block = None if self.representation == 'netlist' else self._extract_features_section(fallback_obs)
        if not features_block:
            try:
                rebuilt_obs = self.obs_builder.build_observation(
                    netlist_text=construct_observation_netlist(self.task_data, netlist_text or self.netlist_text),
                    metrics=self._metrics_for_observation_builder(metrics),
                    op_data={},
                    step=step,
                    constraint_specs=self._get_obs_constraint_specs(),
                )
                features_block = self._extract_features_section(rebuilt_obs)
            except Exception:
                pass
        if features_block:
            obs_lines.append(features_block)
        return "\n".join(obs_lines)

    def _build_initial_observation(
        self,
        initial_metrics: Dict[str, Any],
        initial_obs_xml: Optional[str],
        initial_op_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        initial_op_data = initial_op_data or {}

        # Always re-rendered from the run-time re-simulation; the static dataset text is only the fallback
        # when it fails.
        needs_rebuild = True

        if not initial_op_data:
            sim_result = self._simulate_initial_operating_point(max_attempts=3)
            if sim_result and sim_result.get('op_data'):
                initial_op_data = sim_result.get('op_data', {})
                initial_metrics = {
                    **initial_metrics,
                    'gain_dB': sim_result.get('gain', initial_metrics.get('gain_dB', 0.0)),
                    'power_uW': float(sim_result['power']) * 1e6 if sim_result.get('power') is not None else float('nan'),
                    'bandwidth_MHz': sim_result.get('bandwidth_MHz'),
                    'bandwidth_kind': sim_result.get('bandwidth_kind'),
                    'inband_flatness_db': sim_result.get('inband_flatness_db'),
                    'phase_margin_status': sim_result.get('phase_margin_status'),
                    'phase_margin_deg': sim_result.get('phase_margin_deg', initial_metrics.get('phase_margin_deg')),
                    'differential_gain_db': sim_result.get('differential_gain_db', initial_metrics.get('differential_gain_db')),
                    'offset_v': sim_result.get('offset_v', initial_metrics.get('offset_v')),
                    'cmrr_db': sim_result.get('cmrr_db', initial_metrics.get('cmrr_db')),
                    'psrr_plus_db': sim_result.get('psrr_plus_db', initial_metrics.get('psrr_plus_db')),
                    'psrr_minus_db': sim_result.get('psrr_minus_db', initial_metrics.get('psrr_minus_db')),
                }
            elif sim_result and sim_result.get('error_type') == 'spectre_crash':
                raise RuntimeError(
                    "Initial OP rebuild crashed in Spectre: "
                    f"{sim_result.get('feedback')}"
                )

        obs_metrics = {
            'gain': float(initial_metrics.get('gain_dB', 0.0) or 0.0),
            'power': float(initial_metrics.get('power_uW', 0.0) or 0.0) / 1e6,
            'bandwidth_MHz': initial_metrics.get('bandwidth_MHz'),  # None keeps its meaning
            'bandwidth_kind': initial_metrics.get('bandwidth_kind'),
            'inband_flatness_db': initial_metrics.get('inband_flatness_db'),
            'phase_margin_status': initial_metrics.get('phase_margin_status'),
            'phase_margin_deg': initial_metrics.get('phase_margin_deg'),
            'differential_gain_db': initial_metrics.get('differential_gain_db'),
            'offset_v': initial_metrics.get('offset_v'),
            'cmrr_db': initial_metrics.get('cmrr_db'),
            'psrr_plus_db': initial_metrics.get('psrr_plus_db'),
            'psrr_minus_db': initial_metrics.get('psrr_minus_db'),
        }
        return self.obs_builder.build_observation(
            netlist_text=construct_observation_netlist(self.task_data, self.netlist_text),
            metrics=obs_metrics,
            op_data=self._normalize_op_data_for_observation(initial_op_data),
            step=0,
            constraint_specs=self._get_obs_constraint_specs(),
        )
    
    def step(self, action: Dict) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Apply the action (projection output, possibly with an 'actions' list) and simulate.

        Format/semantic errors are rejected before simulation, physical errors after it.
        Returns (observation, reward, done, info).
        """
        self.t += 1
        
        # Pre-action metrics (passed to the reward computer as info['metrics_pre'])
        self._step_metrics_pre = dict(self.current_metrics)
        
        is_stop_action = action.get('is_stop_action', False) if isinstance(action, dict) else False
        has_error = isinstance(action, dict) and 'error' in action
        is_valid = not has_error

        # =====================================================================
        # TIER 1: PRE-SIMULATION VALIDATION (Format/Semantic)
        # =====================================================================
        
        # Handle action with parsing errors (check for 'error' field from projection)
        if action is not None and isinstance(action, dict) and 'error' in action:
            parse_error = action['error']
            # Never feed unicode/non-English into next-step observations: use ASCII-only message
            if isinstance(parse_error, str) and not parse_error.isascii():
                parse_error = "The content in <think></think> tags must be in English only. Please provide your reasoning in English."
            
            # performance_success from the last valid state (self.current_metrics)
            goal_met_before_error = check_goal_met(
                self.current_metrics,  # Last valid state
                self.goal,
                initial_gain_dB=self.initial_gain_dB
            )
            
            termination_goal_met_before_error = self._resolve_termination_goal_met(
                self.current_metrics,
                goal_met_before_error,
            )
            done = self._check_termination(termination_goal_met_before_error)
            
            # Compute perf_score from last valid metrics for reward computation
            perf_score = self._compute_performance_score(self.current_metrics)
            
            # Compute stop reason
            stop_reason = self._compute_stop_reason(
                done=done,
                goal_met=goal_met_before_error,
                is_stop_action=is_stop_action,
                error_type='parsing_error'
            )
            
            extra: Dict[str, Any] = {
                'performance_success': 1 if goal_met_before_error else 0,
                'error': parse_error, 'error_type': 'parsing_error',
                'stop_reason': stop_reason,
                'action': f'Parse Error: {parse_error}',
            }
            info = self._build_step_info(
                done=done,
                is_action_valid=0, perf_score_post=perf_score,
                extra=extra,
            )
            reward = self._compute_step_reward(info)
            
            # CRITICAL: Update prev_perf_score even for invalid actions so next step has correct perf_score_pre
            self.prev_perf_score = perf_score
            
            # Update goal tracking (use previous step's metrics for error cases)
            self.goal_met_in_prev_step = goal_met_before_error
            
            # Return an error observation so the agent sees the parse error in history.
            obs_lines = self._build_performance_block(
                status="INVALID - parsing_error",
                error=parse_error,
                metrics=self.current_metrics,
            )
            last_valid_features = self._extract_features_section(self._get_last_observation())
            if last_valid_features:
                obs_lines.append(last_valid_features)
            obs = self._cache_observation("\n".join(obs_lines))
            return obs, float(reward), bool(done), info

        
        # =====================================================================
        # HANDLE STOP ACTIONS BEFORE REGULAR VALIDATION
        # =====================================================================
        # Stop actions carry no target/param: skip validation and check the goal directly
        if is_stop_action:
            # Check if goal is actually met
            goal_met = check_goal_met(
                self.current_metrics,  # Current state
                self.goal,
                initial_gain_dB=self.initial_gain_dB
            )
            
            if not goal_met:
                # INCORRECT STOP: the agent declared success but the goal is not met -> penalty, episode ends
                
                # Compute perf_score from current metrics
                perf_score = self._compute_performance_score(self.current_metrics)
                
                done = True
                
                # Build observation with feedback
                obs_lines = self._build_performance_block(
                    status="INVALID - incorrect_stop",
                    error="You output a stop action, but the goal has not been met yet.",
                    metrics=self.current_metrics,
                )
                
                # Add FEATURES section if we have a valid previous observation
                last_valid_features = self._extract_features_section(self._get_last_observation())
                if last_valid_features:
                    obs_lines.append(last_valid_features)
                
                obs = "\n".join(obs_lines)
                
                # Compute stop reason
                stop_reason = self._compute_stop_reason(
                    done=done,
                    goal_met=False,
                    is_stop_action=True,
                    error_type='incorrect_stop'
                )
                
                stop_bonus_penalty = self.reward_config.stop_incorrect_penalty
                extra: Dict[str, Any] = {
                    'error': 'Incorrect stop: goal not met',
                    'error_type': 'incorrect_stop',
                    'stop_reason': stop_reason,
                    'action': 'success (incorrect)',
                    'performance_success': 0,
                }
                info = self._build_step_info(
                    done=done,
                    is_action_valid=0, perf_score_post=perf_score,
                    extra=extra,
                )
                reward = self._compute_step_reward(info) + stop_bonus_penalty
                
                # Update prev_perf_score for next step
                self.prev_perf_score = perf_score
                
                # Update goal tracking
                self.goal_met_in_prev_step = False
                
                self._cache_observation(obs)
                return obs, float(reward), bool(done), info
            
            # A correct stop continues through the normal flow
        
        # a single action dict, or several in its 'actions' list
        action_list = action.get('actions')
        if action_list is None:
            action_list = [action]
        is_valid = True
        error_msg = None
        action_change_mode = getattr(self.reward_config, 'action_change_mode', 'percentage')
        if action_change_mode == 'percentage':
            for a in action_list:
                if not isinstance(a, dict) or a.get('is_stop_action'):
                    continue
                if 'value' in a and not a.get('direction'):
                    continue   # an assignment action: the validator below rejects it by name
                canonical_target, canonical_error = self.action_validator.resolve_target_id(a.get('target_id', ''))
                if canonical_error is not None:
                    is_valid = False
                    error_msg = canonical_error
                    break
                if canonical_target is not None:
                    a['target_id'] = canonical_target
        if is_valid:
            for a in action_list:
                if not isinstance(a, dict) or a.get('is_stop_action'):
                    continue
                is_valid, error_msg = self.action_validator.is_valid_action(a)
                if not is_valid:
                    break
        
        if not is_valid:
            # Format/semantic error - return immediately WITHOUT simulation
            
            # performance_success from the last valid state (self.current_metrics)
            goal_met_before_error = check_goal_met(
                self.current_metrics,  # Last valid state
                self.goal,
                initial_gain_dB=self.initial_gain_dB
            )
            
            termination_goal_met_before_error = self._resolve_termination_goal_met(
                self.current_metrics,
                goal_met_before_error,
            )
            done = self._check_termination(termination_goal_met_before_error)
            
            # Compute perf_score from last valid metrics for reward computation
            perf_score = self._compute_performance_score(self.current_metrics)
            
            # Format action for logging
            action_str = self._format_action_for_log(action)
            
            precheck_error_type = 'validation_failed'
            # Error feedback plus the last FEATURES section
            obs_lines = self._build_performance_block(
                status=f"INVALID - {precheck_error_type}",
                error=error_msg,
                metrics=self.current_metrics,
            )
            
            # Add FEATURES section if we have a valid previous observation
            last_valid_features = self._extract_features_section(self._get_last_observation())
            if last_valid_features:
                obs_lines.append(last_valid_features)
            
            obs = "\n".join(obs_lines)
            
            # Compute stop reason
            stop_reason = self._compute_stop_reason(
                done=done,
                goal_met=termination_goal_met_before_error,
                is_stop_action=is_stop_action,
                error_type=precheck_error_type
            )
            
            extra: Dict[str, Any] = {
                'performance_success': 1 if goal_met_before_error else 0,
                'error': error_msg, 'error_type': precheck_error_type,
                'stop_reason': stop_reason,
                'action': f'Invalid: {action_str}',
            }
            info = self._build_step_info(
                done=done,
                is_action_valid=0, perf_score_post=perf_score,
                extra=extra,
            )
            reward = self._compute_step_reward(info)
            
            # CRITICAL: Update prev_perf_score even for invalid actions so next step has correct perf_score_pre
            self.prev_perf_score = perf_score
            
            # Update goal tracking (use previous step's metrics for error cases)
            self.goal_met_in_prev_step = goal_met_before_error
            
            self._cache_observation(obs)
            return obs, float(reward), bool(done), info
        
        # =====================================================================
        # TIER 2: POST-SIMULATION VALIDATION (Physical)
        # =====================================================================
        # Action format is valid, now apply it and simulate
        
        # Create step directory for this simulation
        step_dir = self.workdir / f"step_{self.t:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply action(s) to netlist: support one or multiple actions in sequence
        action_list = action.get('actions')
        if action_list is None:
            action_list = [action]
        new_netlist_text = self.netlist_text
        all_checks = []
        for a in action_list:
            new_netlist_text, action_checks = self._apply_action_to_netlist(a, new_netlist_text)
            all_checks.extend(action_checks)

        bounds_error = self._get_geometry_bounds_error(new_netlist_text)
        if bounds_error:
            # the out-of-bounds design is committed (the episode continues from it) with empty metrics
            self.netlist_text = new_netlist_text
            self.current_metrics = {}

            goal_met_on_error = check_goal_met(
                self.current_metrics,
                self.goal,
                initial_gain_dB=self.initial_gain_dB
            )
            termination_goal_met_on_error = self._resolve_termination_goal_met(
                self.current_metrics,
                goal_met_on_error,
            )
            done = self._check_termination(termination_goal_met_on_error)
            perf_score = 0.0
            action_str = self._format_action_for_log(action)
            precheck_error_type = 'geometry_bounds_error'
            obs = self._build_design_error_observation(
                error_msg=bounds_error,
                metrics=self.current_metrics,
                step=self.t,
                netlist_text=self.netlist_text,
                status="INVALID - design_error",
                omit_perf_metrics=True,
            )
            stop_reason = self._compute_stop_reason(
                done=done,
                goal_met=termination_goal_met_on_error,
                is_stop_action=is_stop_action,
                error_type=precheck_error_type
            )
            extra: Dict[str, Any] = {
                'performance_success': 1 if goal_met_on_error else 0,
                'error': bounds_error, 'error_type': precheck_error_type,
                'stop_reason': stop_reason, 'action': f'Invalid: {action_str}',
            }
            info = self._build_step_info(
                done=done,
                is_action_valid=0, perf_score_post=perf_score,
                extra=extra,
            )
            reward = self._compute_step_reward(info)
            self.prev_perf_score = perf_score
            self.goal_met_in_prev_step = goal_met_on_error
            self._cache_observation(obs)
            return obs, float(reward), bool(done), info
        
        # Save netlist for simulation
        tmp_netlist = step_dir / "netlist.scs"
        tmp_netlist.write_text(new_netlist_text, encoding='utf-8')
        
        # Simulate
        sim_result = self.simulator.run_simulation(
            netlist_content=new_netlist_text,
            timeout=900,
            threads=1
        )
        
        # Check for simulation failure (valid=True only when gain > 0 in spectre_runner)
        if not sim_result['valid']:
            
            error_type = sim_result.get('error_type', 'simulation_error')
            error_msg = sim_result.get('feedback', 'Unknown error')
            if error_type in {'infrastructure_error', 'python_exception'}:
                raise RuntimeError(f"Simulator infrastructure failure: {error_type}: {error_msg}")
            
            # Actual sim values when available (the sim ran but gain was non-positive)
            _fail_power = sim_result.get('power')
            _fail_bw = sim_result.get('bandwidth_MHz')
            failed_metrics = {
                'gain': sim_result.get('gain'),
                'gain_dB': sim_result.get('gain'),
                'power_uW': (_fail_power * 1e6) if _fail_power is not None else None,
                'bandwidth_MHz': _fail_bw,
                'bandwidth_kind': sim_result.get('bandwidth_kind'),
                'inband_flatness_db': sim_result.get('inband_flatness_db'),
                'phase_margin_deg': sim_result.get('phase_margin_deg'),
                'phase_margin_status': sim_result.get('phase_margin_status'),
                'differential_gain_db': sim_result.get('differential_gain_db'),
                'offset_v': sim_result.get('offset_v'),
                'cmrr_db': sim_result.get('cmrr_db'),
                'psrr_plus_db': sim_result.get('psrr_plus_db'),
                'psrr_minus_db': sim_result.get('psrr_minus_db'),
            }
            # A simulation that returned a gain (e.g. negative) is a real state: commit netlist + measured
            # metrics. Without a gain (convergence failure / no output) roll back: netlist and metrics stay at
            # the last valid state and the observation reports it alongside the error.
            _fail_gain_val = failed_metrics.get('gain_dB')
            _sim_returned_gain = (
                _fail_gain_val is not None
                and math.isfinite(float(_fail_gain_val))
            )
            _commit_failed_state = _sim_returned_gain
            if _commit_failed_state:
                _gain_nonpositive = float(_fail_gain_val) <= 0.0
                committed_metrics = dict(failed_metrics)
                # (value, status) must travel as a pair: the pm / bandwidth value and status keys are
                # taken together from this simulation; if missing, both are cleared, never back-filled.
                if _gain_nonpositive:
                    committed_metrics['bandwidth_MHz'] = None
                    committed_metrics['bandwidth_kind'] = None
                obs_metrics = committed_metrics
            else:
                obs_metrics = self.current_metrics
            goal_met_on_failure = check_goal_met(
                obs_metrics,
                self.goal,
                initial_gain_dB=self.initial_gain_dB
            )
            perf_score = self._compute_performance_score(obs_metrics)
            obs_lines = self._build_performance_block(
                status=f"INVALID - {error_type}",
                error=error_msg,
                metrics=obs_metrics,
            )
            
            last_valid_features = (
                self._extract_features_section(self._get_last_observation()) if not _commit_failed_state else None
            )
            if last_valid_features:
                obs_lines.append(last_valid_features)
            else:
                try:
                    features_obs = self.obs_builder.build_observation(
                        netlist_text=construct_observation_netlist(
                            self.task_data,
                            new_netlist_text if _commit_failed_state else self.netlist_text,
                        ),
                        metrics=self._metrics_for_observation_builder(obs_metrics),
                        op_data={},
                        step=self.t,
                        constraint_specs=self._get_obs_constraint_specs(),
                    )
                    obs_lines.append(features_obs)
                except Exception as e:
                    raise RuntimeError("Cannot render the retained simulation-failure state") from e
            
            obs = "\n".join(obs_lines)
            
            # Determine if episode should end
            termination_goal_met_on_failure = self._resolve_termination_goal_met(
                obs_metrics,
                goal_met_on_failure,
            )
            done = self._check_termination(termination_goal_met_on_failure)
            
            # Compute stop reason
            stop_reason = self._compute_stop_reason(
                done=done,
                goal_met=termination_goal_met_on_failure,
                is_stop_action=is_stop_action,
                error_type=error_type
            )
            
            extra: Dict[str, Any] = {
                'performance_success': 1 if goal_met_on_failure else 0,
                'error': error_msg, 'error_type': error_type,
                'stop_reason': stop_reason,
                'simulation_measurement_valid': bool(_sim_returned_gain),
                'state_transition': 'committed_measured_failure' if _commit_failed_state else 'rolled_back_simulation_failure',
            }
            if _commit_failed_state:
                self.netlist_text = new_netlist_text
                self.current_metrics = committed_metrics

            info = self._build_step_info(
                metrics=obs_metrics,
                done=done,
                is_action_valid=0, perf_score_post=perf_score,
                extra=extra,
            )
            reward = self._compute_step_reward(info)
            
            # CRITICAL: Update prev_perf_score even for failed simulations so next step has correct perf_score_pre
            self.prev_perf_score = perf_score
            
            # Update goal tracking with the state used for this failed step.
            self.goal_met_in_prev_step = goal_met_on_failure
            
            self._cache_observation(obs)
            return obs, float(reward), bool(done), info
        
        # Extract metrics from simulation (convert to expected format)
        metrics_post = {
            'gain': sim_result['gain'],        # dB
            'power': sim_result['power'],      # Watts (will be converted to uW in observation)
            'gain_dB': sim_result['gain'],     # For compatibility
            'differential_gain_db': sim_result.get('differential_gain_db'),
            'power_uW': sim_result['power'] * 1e6 if sim_result['power'] is not None else float('nan'),
            'bandwidth_MHz': sim_result.get('bandwidth_MHz'),  # MHz everywhere
            'bandwidth_kind': sim_result.get('bandwidth_kind'),
            'inband_flatness_db': sim_result.get('inband_flatness_db'),
            'phase_margin_status': sim_result.get('phase_margin_status'),
            'phase_margin_deg': sim_result.get('phase_margin_deg'),
            'offset_v': sim_result.get('offset_v'),
            'cmrr_db': sim_result.get('cmrr_db'),
            'psrr_plus_db': sim_result.get('psrr_plus_db'),
            'psrr_minus_db': sim_result.get('psrr_minus_db'),
        }
        
        op_data = sim_result.get('op_data', {})
        perf_score_post = self._compute_performance_score(metrics_post)
        
        # Save old perf score BEFORE updating
        perf_score_pre = self.prev_perf_score
        
        # Check goal achievement FIRST (before updating tracking)
        goal_met = check_goal_met(metrics_post, self.goal, initial_gain_dB=self.initial_gain_dB)
        termination_goal_met = self._resolve_termination_goal_met(metrics_post, goal_met)
        
        # Update tracking for NEXT step
        self.prev_perf_score = perf_score_post
        
        # Update current metrics for next error handling
        _bw_post = metrics_post.get('bandwidth_MHz')
        self.current_metrics = {
            'gain_dB': float(metrics_post.get('gain_dB', 0.0)),
            'power_uW': float(metrics_post.get('power_uW', 0.0)),
            # None = not measurable and must be kept (float(None) would crash, 0.0 would fake a baseline)
            'bandwidth_MHz': float(_bw_post) if _bw_post is not None else None,
            'bandwidth_kind': metrics_post.get('bandwidth_kind'),
            'inband_flatness_db': metrics_post.get('inband_flatness_db'),
            'phase_margin_status': metrics_post.get('phase_margin_status'),
            'phase_margin_deg': metrics_post.get('phase_margin_deg'),  # None if not from sim
            'differential_gain_db': metrics_post.get('differential_gain_db'),
            'offset_v': metrics_post.get('offset_v'),
            'cmrr_db': metrics_post.get('cmrr_db'),
            'psrr_plus_db': metrics_post.get('psrr_plus_db'),
            'psrr_minus_db': metrics_post.get('psrr_minus_db'),
        }
        self._maybe_update_best_achieved(self.current_metrics, perf_score_post=perf_score_post)
        
        # Determine termination
        done = self._check_termination(termination_goal_met)
        
        # Compute reward using integrated reward computer.
        info_for_reward = self._build_step_info(
            metrics=metrics_post,
            done=done,
            is_action_valid=1,
            perf_score_post=perf_score_post,
            perf_score_pre=perf_score_pre,
            extra={
                'performance_success': bool(goal_met),
                'is_stop_action': is_stop_action,
            },
        )
        reward = self._compute_step_reward(info_for_reward)
        
        # Update state for next step
        self.netlist_text = new_netlist_text
        
        # Build observation (only FEATURES section - the changing part); use circuit-only so testbench is hidden
        obs = self.obs_builder.build_observation(
            netlist_text=construct_observation_netlist(self.task_data, new_netlist_text),
            metrics=metrics_post,
            op_data=self._normalize_op_data_for_observation(op_data),
            step=self.t,
            constraint_specs=self._get_obs_constraint_specs(),
        )
        # Cache valid observation for invalid action handling
        self._cache_observation(obs, valid=True)
        
        # Compute stop reason
        stop_reason = self._compute_stop_reason(
            done=done,
            goal_met=termination_goal_met,
            is_stop_action=is_stop_action,
            error_type=None
        )
        
        # Build info dict
        info = self._build_step_info(
            metrics=metrics_post,
            extra={
                'done': bool(done),
                'performance_success': 1 if goal_met else 0,
                'perf_score_post': perf_score_post,
                'perf_score_pre': perf_score_pre,  # Include for logging (saved before the update above)
                'perf_score_initial': float(getattr(self, 'initial_perf_score', 0.0)),
                'is_action_valid': 1,  # Valid action (passed semantic validation and simulation)
                'is_stop_action': is_stop_action,
                'stop_reason': stop_reason,
                'already_achieved_goal': self.goal_met_in_prev_step,  # For further-exploration post-goal reward
            },
        )
        
        # Update goal tracking for the next step
        self.goal_met_in_prev_step = goal_met
        
        return obs, float(reward), bool(done), info
    
    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================
    
    def format_goal_for_display(self, goal_dict: Dict) -> str:
        """Format goal for feedback messages."""
        if not goal_dict:
            return "optimize performance"
        
        mode = goal_dict.get('mode', 'constraint')
        if mode == 'optimization':
            return "maximize gain and minimize power"
        else:
            constraint_specs = goal_dict.get('constraint_specs')
            if constraint_specs is not None:
                constraint_specs = set(constraint_specs)

            def should_show(spec_name: str, value) -> bool:
                if value is None:
                    return False
                if isinstance(value, (int, float)) and spec_name != 'offset' and value <= 0:
                    return False
                if constraint_specs is None:
                    return True
                return spec_name in constraint_specs

            parts = []

            target_gain = goal_dict.get('gain_min')
            if should_show('gain', target_gain):
                parts.append(f"gain >= {target_gain:.1f} dB")

            power_budget = goal_dict.get('power_max')
            if should_show('power', power_budget):
                power_budget = float(power_budget)
                if power_budget < 1.0:
                    power_text = f"{power_budget:.3f}".rstrip('0').rstrip('.')
                else:
                    power_text = f"{power_budget:.1f}"
                parts.append(f"power <= {power_text} uW")

            bandwidth_min = goal_dict.get('bandwidth_min')
            if should_show('bandwidth', bandwidth_min):
                parts.append(f"-3dB bandwidth >= {self._format_bw_mhz_3sig(bandwidth_min)}")

            phase_margin_min_deg = goal_dict.get('phase_margin_min_deg')
            if should_show('phase_margin', phase_margin_min_deg):
                parts.append(f"phase_margin >= {phase_margin_min_deg:.1f} deg")

            offset_abs_max_v = goal_dict.get('offset_abs_max_v')
            if should_show('offset', offset_abs_max_v):
                parts.append(f"|offset| <= {offset_abs_max_v * 1e3:.2f} mV")

            cmrr_min_db = goal_dict.get('cmrr_min_db')
            if should_show('cmrr', cmrr_min_db):
                parts.append(f"CMRR >= {cmrr_min_db:.1f} dB")

            psrr_plus_min_db = goal_dict.get('psrr_plus_min_db')
            if should_show('psrr_plus', psrr_plus_min_db):
                parts.append(f"PSRR+ >= {psrr_plus_min_db:.1f} dB")

            psrr_minus_min_db = goal_dict.get('psrr_minus_min_db')
            if should_show('psrr_minus', psrr_minus_min_db):
                parts.append(f"PSRR- >= {psrr_minus_min_db:.1f} dB")

            return (", ".join(parts) + _tolerance_note_for_goal(goal_dict)) if parts else "optimize performance"

    @staticmethod
    def _format_action_for_log(action: Dict[str, Any]) -> str:
        """Format percentage or absolute assignment actions for logs/errors."""
        action_list = action.get('actions') if isinstance(action, dict) else None
        if action_list is None and isinstance(action, dict):
            action_list = [action]
        if not isinstance(action_list, list):
            return str(action)

        rendered = []
        for a in action_list:
            if not isinstance(a, dict):
                rendered.append(str(a))
                continue
            if a.get('is_stop_action'):
                rendered.append("success")
                continue
            if 'direction' in a and 'percentage' in a:
                rendered.append(
                    f"{a.get('direction', 'N/A')} {a.get('target_id', 'N/A')} "
                    f"{a.get('param', 'N/A')} {a.get('percentage', 0)}%"
                )
                continue
            if 'param' in a and ('value' in a or 'to_value' in a):
                rendered.append(f"{a.get('param', 'N/A')}={a.get('value', a.get('to_value', 'N/A'))}")
                continue
            rendered.append(str(a))
        return "; ".join(rendered) if rendered else str(action)
    
    def _apply_action_to_netlist(self, action: Dict, netlist: str) -> Tuple[str, List[Dict]]:
        """Apply a percentage action {direction, target_id, param, percentage} via NetlistActionTool; returns
        (new_netlist, action_checks)."""
        try:
            # Parse action dict to normalized action dicts (requires netlist for current values)
            action_dicts = self.action_tool.parse_actions(action, netlist_text=netlist)
            
            # Apply actions to netlist
            new_netlist, checks = self.action_tool.apply_actions(netlist, action_dicts)
            
            return new_netlist, checks
            
        except (ValueError, json.JSONDecodeError) as e:
            # Action parsing failed - return original netlist with error info
            return netlist, [{
                'applied': False,
                'note': f"Action parsing failed: {str(e)}"
            }]
    
    def _check_termination(self, goal_met: bool) -> bool:
        """A met constraint goal ends the episode; with allow_further_exploration_after_goal it never does (the rollout
        loop's turn limit ends it). Otherwise the episode ends at max_steps."""
        if self.goal.get('mode', 'constraint') == 'constraint' and goal_met:
            return not self._goal_allows_further_exploration_after_goal()
        return self.t >= self.max_steps

    def _compute_stop_reason(self, done: bool, goal_met: bool, is_stop_action: bool, error_type: str = None) -> str:
        """Termination reason, or None while the episode continues."""
        if not done:
            return None

        # Check for error-based termination first
        if error_type:
            if error_type in ['parsing_error', 'validation_failed', 'geometry_bounds_error', 'design_error', 'convergence_failed']:
                return 'forced_termination'
        
        # Check for correct stop
        if is_stop_action and goal_met:
            return 'correct_stop'
        
        # Check for max steps
        if self.t >= self.max_steps:
            if goal_met:
                return 'max_steps_goal_met'
            else:
                return 'max_steps'
        
        # Goal met (auto termination)
        if goal_met:
            return 'goal_met'
        
        # Default
        return 'unknown'
