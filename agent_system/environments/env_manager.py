# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any, Optional
from collections import defaultdict
import torch
import numpy as np

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

from functools import partial
import os
import re
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class CircuitopEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for Circuitop environment.
    Extracts the goal, builds prompts with history, stores to memory.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        
        # Load reward configuration from config
        from agent_system.environments.env_package.circuitop.reward_config import load_reward_config
        reward_config_spec = config.env.circuitop.get('reward_config', None)
        self.reward_config = load_reward_config(reward_config_spec)
        
        # Storage for circuit structure and metadata (persistent across episode)
        self.circuit_structures = []  # Circuit topology/netlist per env
        self.topologies = []  # Topology names per env
    
    def sync_group_states(self, selected_indices, group_size, last_obs, skip_groups=None):
        """SLIC continuation sync without re-simulation.

        Copies each group's selected branch state onto its peers: worker env
        state, memory rows, pre_text_obs, admissible commands and the latest
        observation entries. Non-selected branches are overwritten (their
        training rows were already harvested).

        skip_groups: optional per-group bool mask; True = the group's episode
        is done, leave its envs untouched (they drift past the episode end and
        their later rows are padding).
        """
        import copy
        n = len(self.pre_text_obs)
        sel = [int(s) for s in selected_indices]
        if skip_groups is not None:
            sel = [-1 if skip_groups[g] else s for g, s in enumerate(sel)]
        live = [(g, s) for g, s in enumerate(sel) if s >= 0]
        states = self.envs.get_states([s for _, s in live])
        state_by_group = {g: st for (g, _), st in zip(live, states)}
        assignments = []
        for g, sidx in live:
            state = state_by_group[g]
            for i in range(g * group_size, min((g + 1) * group_size, n)):
                if i != sidx:
                    assignments.append((i, state))
        if assignments:
            self.envs.set_states(assignments)
        # manager-side per-env state (skipped groups: sidx = -1)
        for g, sidx in enumerate(sel):
            if sidx < 0:
                continue
            for i in range(g * group_size, min((g + 1) * group_size, n)):
                if i == sidx:
                    continue
                self.memory._data[i] = copy.deepcopy(self.memory._data[sidx])
                self.pre_text_obs[i] = self.pre_text_obs[sidx]
                if getattr(self.envs, "prev_admissible_commands", None) is not None:
                    self.envs.prev_admissible_commands[i] = copy.deepcopy(
                        self.envs.prev_admissible_commands[sidx])
        # observation entries: peers adopt the selected branch's obs
        obs = {}
        for key, value in last_obs.items():
            if isinstance(value, list) and len(value) == n:
                new_list = list(value)
                for g, sidx in enumerate(sel):
                    if sidx < 0:
                        continue
                    for i in range(g * group_size, min((g + 1) * group_size, n)):
                        if i != sidx:
                            new_list[i] = value[sidx]
                obs[key] = new_list
            else:
                obs[key] = value
        return obs
    # ---- MT-GRPO tree: env state capture / restore ----------------------
    # Used only by mt_grpo_tree_rollout.py: snapshot an env's full state (the same surface as
    # sync_group_states) and restore it later to branch from an interior tree node.

    def tree_capture_env_states(self, indices, last_obs):
        """Snapshot the full per-env state for each index. 0 simulator calls."""
        import copy
        idx = [int(i) for i in indices]
        worker_states = self.envs.get_states(idx)
        snaps = []
        for i, wstate in zip(idx, worker_states):
            snaps.append({
                "worker_state": wstate,
                "memory_row": copy.deepcopy(self.memory._data[i]),
                "pre_text_obs": self.pre_text_obs[i],
                "admissible": (
                    copy.deepcopy(self.envs.prev_admissible_commands[i])
                    if getattr(self.envs, "prev_admissible_commands", None) is not None
                    else None
                ),
                "obs_entry": {
                    key: value[i]
                    for key, value in last_obs.items()
                    if isinstance(value, list) and len(value) == len(self.pre_text_obs)
                },
            })
        return snaps

    def tree_restore_env_states(self, assignments, last_obs):
        """Push captured snapshots back onto envs. 0 simulator calls.

        assignments: list of (env_index, snapshot-from-tree_capture_env_states).
        Returns an obs dict shaped like last_obs with restored entries swapped in.
        """
        import copy
        self.envs.set_states([(int(i), s["worker_state"]) for i, s in assignments])
        for i, s in assignments:
            i = int(i)
            self.memory._data[i] = copy.deepcopy(s["memory_row"])
            self.pre_text_obs[i] = s["pre_text_obs"]
            if (s["admissible"] is not None
                    and getattr(self.envs, "prev_admissible_commands", None) is not None):
                self.envs.prev_admissible_commands[i] = copy.deepcopy(s["admissible"])
        obs = {}
        n = len(self.pre_text_obs)
        for key, value in last_obs.items():
            if isinstance(value, list) and len(value) == n:
                new_list = list(value)
                for i, s in assignments:
                    if key in s["obs_entry"]:
                        new_list[int(i)] = s["obs_entry"][key]
                obs[key] = new_list
            else:
                obs[key] = value
        return obs

    def reset(self, kwargs):
        # kwargs is a list/array of dicts, one per environment
        text_obs, infos = self.envs.reset(kwargs)
        
        # Initialize memory
        self.memory.reset(batch_size=len(text_obs))
        self.pre_text_obs = text_obs
        
        # Per-env goal, circuit structure and topology
        self.goals = []
        self.circuit_structures = []
        self.topologies = []
        
        for i in range(len(text_obs)):
            env_kwargs = kwargs[i] if kwargs is not None and i < len(kwargs) else {}
            task_data = env_kwargs.get('task_data', {})
            
            # Extract circuit structure for system messages (topology description)
            # Use circuit-only so testbench is not shown
            # same rule as CircuitopWorker.reset: env.circuitop.representation when set, else the row's, else an error
            from agent_system.environments.env_package.circuitop.circuitop.representation import resolve_representation
            representation = resolve_representation(self.config.env.circuitop.get('representation'), env_kwargs.get('representation'), task_data)
            if task_data and 'structure' in task_data:
                structure_content = task_data['structure'].get(representation, '')
            else:
                structure_content = ''
            if representation == 'netlist':
                structure_content = None
            else:
                from agent_system.environments.env_package.circuitop.circuitop.state_utils import (
                    format_si,
                    format_tech_geometry_bounds,
                    parse_si,
                )
                # Neither the dataset row nor the driver process carries the tech node, so use
                # env.circuitop.tech_node; if unset, get_tech_node() fails fast.
                tech_node = env_kwargs.get('tech_node')
                if not tech_node:
                    circuitop_cfg = getattr(getattr(self.config, 'env', None), 'circuitop', None)
                    tech_node = circuitop_cfg.get('tech_node') if circuitop_cfg is not None else None

                def _extract_vdd_from_task_data(task_data_obj: Any) -> Optional[float]:
                    if not isinstance(task_data_obj, dict):
                        return None

                    # 1) Primary source: task_data.initial_state.params.vdd
                    params = ((task_data_obj.get('initial_state') or {}).get('params') or {})
                    for key, value in params.items():
                        if str(key).strip().lower() == "vdd" and value is not None:
                            parsed = parse_si(str(value).strip()) if not isinstance(value, (int, float)) else float(value)
                            if np.isfinite(parsed):
                                return float(parsed)
                            return None

                    # 2) Fallback: initial_state.metrics style fields if present.
                    metrics = ((task_data_obj.get('initial_state') or {}).get('metrics') or {})
                    for key in ("vdd", "supply_voltage", "vdd_v", "supply_voltage_v"):
                        raw = metrics.get(key)
                        if raw is None:
                            continue
                        try:
                            parsed = parse_si(str(raw).strip()) if not isinstance(raw, (int, float)) else float(raw)
                        except Exception:
                            continue
                        if np.isfinite(parsed):
                            return float(parsed)

                    # 3) Fallback extraction: parse `parameters ... vdd=...` from structure text.
                    structure = task_data_obj.get('structure') or {}
                    for key in ("netlist",):
                        raw = structure.get(key)
                        if not isinstance(raw, str) or not raw:
                            continue
                        for line in raw.splitlines():
                            line = line.strip()
                            if not line.lower().startswith("parameters "):
                                continue
                            m = re.search(r"(?:^|\\s)vdd\\s*=\\s*([^\\s]+)", line, flags=re.IGNORECASE)
                            if not m:
                                continue
                            parsed = parse_si(m.group(1))
                            if np.isfinite(parsed):
                                return float(parsed)
                    return None

                vdd_value = _extract_vdd_from_task_data(task_data)
                if vdd_value is None:
                    # Prompt-side default when the row carries no vdd
                    vdd_value = 1.2
                vdd_note = f"Fixed global: vdd={format_si(vdd_value)}V (user-set, not tunable)."
                # 65nm per-topology L upper bound: the geometry box shown in the prompt must match the env validation
                bounds_note = ("\n\nDesign bounds:\n"
                               + format_tech_geometry_bounds(tech_node, topology=env_kwargs.get('topology'))
                               + "\n" + vdd_note)
                structure_content = (structure_content or "") + bounds_note
            self.circuit_structures.append(structure_content)
            
            # Extract topology name for per-circuit logging
            topology = env_kwargs.get('topology', 'unknown')
            self.topologies.append(topology)
            
            # Enriched goal (with tolerances) from env info, not the dataset kwargs
            task_goal = infos[i].get('goal', {})
            
            # Ensure task_goal is never None - provide empty dict as fallback
            if task_goal is None:
                task_goal = {}
            
            self.goals.append(task_goal)

        self.system_messages = list(self.circuit_structures)
        full_text_obs = self.build_text_obs(text_obs, init=True)
        
        
        # Return structured observations with system messages
        return {
            'text': full_text_obs, 
            'system': self.system_messages,
            'image': None, 
            'anchor': text_obs
        }, infos
    
    def step(self, text_actions: List[str]):
        # Admissible actions per env (get_admissible_commands is a property, not a callable)
        action_pools = [self.envs.get_admissible_commands[i] for i in range(len(text_actions))]
        raw_action_outputs = [action if isinstance(action, str) else str(action) for action in text_actions]
        
        # Project text actions to action dicts
        actions, valids = self.projection_f(text_actions, action_pools)
        
        # Execute actions in environment
        text_obs, rewards, dones, infos = self.envs.step(actions)

        # Optional: treat certain projection format errors as terminal (end episode immediately)
        circuitop_cfg = getattr(self.config.env, "circuitop", None)
        if circuitop_cfg is not None:
            # 1) Trailing-content errors after final </action>
            end_on_trailing_content = bool(circuitop_cfg.get('end_on_trailing_content', False))
            # 2) Prefix errors: first token not <think>
            end_on_think_prefix_error = bool(circuitop_cfg.get('end_on_think_prefix_error', False))
            # 3) Any invalid format/validation (not simulation failure): end episode + trainer applies invalid_action_penalty_coef
            end_on_invalid_format = bool(circuitop_cfg.get('end_on_invalid_format', False))

            trailing_error_msg = 'Your response must end immediately after the final </action> tag with no extra text or symbols.'
            think_prefix_error_msg = 'Response must begin with <think>...</think> reasoning block before any other text.'
            # Format/validation error types (exclude simulation/design/rejection)
            FORMAT_ERROR_TYPES = frozenset({'parsing_error', 'validation_failed', 'incorrect_stop'})

            if end_on_trailing_content or end_on_think_prefix_error or end_on_invalid_format:
                for i, info in enumerate(infos):
                    if not isinstance(info, dict):
                        continue
                    should_end = False
                    if end_on_invalid_format:
                        et = info.get('error_type', '')
                        if et in FORMAT_ERROR_TYPES:
                            should_end = True
                    if not should_end:
                        err = info.get('error', '')
                        if end_on_trailing_content and err == trailing_error_msg:
                            should_end = True
                        if end_on_think_prefix_error and err == think_prefix_error_msg:
                            should_end = True
                    if should_end:
                        dones[i] = True
                        info['done'] = True
        
        # Annotate action dicts with env-side errors (validation_failed, design_error)
        # so history shows both the attempted action AND why it failed.
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            env_err = info.get('error', '')
            if env_err and isinstance(actions[i], dict) and 'error' not in actions[i]:
                actions[i]['env_error'] = env_err

        # Store previous observation, parsed action dicts, and raw model output to memory.
        # History rendering can then choose DSL-only summaries or the exact decoded output.
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'raw_action_output': raw_action_outputs})
        self.pre_text_obs = text_obs

        # Build full text observation with history
        full_text_obs = self.build_text_obs(text_obs)

        # Action validity for the trainer's invalid-action penalty; a value set by the env wins
        for i, info in enumerate(infos):
            if 'is_action_valid' not in info:
                info['is_action_valid'] = valids[i]

        next_observations = {
            'text': full_text_obs,
            'system': getattr(self, 'system_messages', self.circuit_structures),
            'image': None,
            'anchor': text_obs
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def format_goal_for_display(self, goal_dict: Dict) -> str:
        """Goal dict as prompt text (normalization parameters hidden)."""
        if not goal_dict:
            return "Optimize circuit performance"

        mode = goal_dict.get('mode', 'constraint')
        
        if mode == 'optimization':
            # Optimization mode: show objective only, hide normalization parameters
            objective = goal_dict.get('objective', 'max_gain_min_power')
            if objective == 'max_gain_min_power':
                return "Maximize Gain and Minimize Power"
            else:
                return f"Optimize: {objective}"
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
                # adaptive unit at 3 sig digits — f3db anchors span kHz..hundreds of MHz
                _hz = float(bandwidth_min) * 1e6
                for _scale, _unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")):
                    if abs(_hz) >= _scale or _unit == "Hz":
                        parts.append(f"-3dB bandwidth >= {_hz / _scale:.3g} {_unit}")
                        break

            phase_margin_min_deg = goal_dict.get('phase_margin_min_deg')
            phase_margin_max_deg = goal_dict.get('phase_margin_max_deg')
            if should_show('phase_margin', phase_margin_min_deg):
                if phase_margin_max_deg is not None:
                    parts.append(f"{phase_margin_min_deg:.1f} deg <= phase_margin <= {phase_margin_max_deg:.1f} deg")
                else:
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

            if not parts:
                return "Optimize circuit performance"
            return "Achieve " + ", ".join(parts) + _tolerance_note_for_goal(goal_dict)
    
    def extract_goal(self, infos: List[Dict]) -> List[str]:
        """
        Extract goal description from infos.
        Handles both constraint and optimization modes from dataset.
        """
        goals = []
        for info in infos:
            goal_dict = info.get('goal', {})
            goals.append(self.format_goal_for_display(goal_dict))
        return goals
    
    def _format_action_as_dsl(self, action) -> str:
        """
        Convert action dict or string to DSL format for display in history.
        Ensures feedback never includes raw unicode/non-English: only ASCII error messages
        are shown so the next-step observation does not contain the model's unicode output.
        """
        import ast

        _INVALID_NON_ENGLISH = "Invalid action (non-English or unicode content)."

        # Handle dict objects directly (the common case)
        if isinstance(action, dict):
            # Check for projection-level error first (parsing/format errors)
            if 'error' in action:
                err = action['error']
                if isinstance(err, str) and not err.isascii():
                    err = _INVALID_NON_ENGLISH
                proposed = action.get('proposed_action_raw', '')
                if proposed and isinstance(proposed, str) and proposed.isascii():
                    return f"{proposed} [Error: {err}]"
                return f"Error: {err}"
            # Check for invalid marker
            if action is None or action.get('target_id') == 'invalid':
                return "Invalid action"
            # Build DSL text from action fields
            env_err = action.get('env_error', '')
            # Handle multi-action wrapper: {'actions': [dict, ...]}
            if 'actions' in action and isinstance(action['actions'], list):
                parts = []
                for a in action['actions']:
                    if isinstance(a, dict) and 'direction' in a and 'percentage' in a:
                        d = a.get('direction', '')
                        t = a.get('target_id', '')
                        p = a.get('param', '')
                        pct = a.get('percentage', 0)
                        parts.append(f"{d} {t} {p} {pct}%")
                    elif isinstance(a, dict) and 'param' in a and ('value' in a or 'to_value' in a):
                        p = a.get('param', '')
                        v = a.get('value', a.get('to_value', ''))
                        parts.append(f"{p}={v}")
                    else:
                        parts.append(str(a))
                dsl = "; ".join(parts) if parts else "Invalid action"
            else:
                # Format single valid action (DSL is ASCII)
                if 'direction' in action and 'percentage' in action:
                    direction = action.get('direction', '')
                    target_id = action.get('target_id', '')
                    param = action.get('param', '')
                    percentage = action.get('percentage', 0)
                    dsl = f"{direction} {target_id} {param} {percentage}%"
                elif 'param' in action and ('value' in action or 'to_value' in action):
                    param = action.get('param', '')
                    value = action.get('value', action.get('to_value', ''))
                    dsl = f"{param}={value}"
                else:
                    dsl = str(action)
            if env_err and isinstance(env_err, str) and env_err.isascii():
                return f"{dsl} [Error: {env_err}]"
            return dsl
        
        # Handle a string representation: never show raw string if it contains unicode
        if isinstance(action, str):
            if not action.isascii():
                return _INVALID_NON_ENGLISH
            try:
                action_dict = ast.literal_eval(action)
                if isinstance(action_dict, dict):
                    if 'direction' in action_dict and 'percentage' in action_dict:
                        direction = action_dict.get('direction', '')
                        target_id = action_dict.get('target_id', '')
                        param = action_dict.get('param', '')
                        percentage = action_dict.get('percentage', 0)
                        return f"{direction} {target_id} {param} {percentage}%"
                    if 'param' in action_dict and ('value' in action_dict or 'to_value' in action_dict):
                        param = action_dict.get('param', '')
                        value = action_dict.get('value', action_dict.get('to_value', ''))
                        return f"{param}={value}"
            except (ValueError, SyntaxError):
                pass
            return action
        
        # Unknown type: stringify but sanitize if non-ASCII
        s = str(action)
        return _INVALID_NON_ENGLISH if not s.isascii() else s

    @staticmethod
    def _circuitop_template(goal: dict, with_history: bool) -> str:
        """The prompt for this goal: the reward preset's goal_prompt_style picks one of the two released prompts
        (agent_system/environments/prompts/circuitop.py); the first turn uses the form without history."""
        style = goal.get('prompt_goal_style')
        if style not in CIRCUITOP_TEMPLATES:
            raise ValueError(f"Unknown goal_prompt_style {style!r}; released prompts: {sorted(CIRCUITOP_TEMPLATES)}.")
        return CIRCUITOP_TEMPLATES[style][1 if with_history else 0]

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        Build the text observation for the agent.
        Formats prompt with goal/history/current observation.
        """
        postprocess_text_obs = []
        admissible_actions_by_env = []
        
        # Fetch memory if history is enabled and not initial step
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )
            
            formatted_memory_contexts = []
            for i in range(len(memory_contexts)):
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_len = len(recent_history)
                start_idx = len(self.memory[i]) - valid_len
                
                lines = []
                for j, rec in enumerate(recent_history):
                    step_num = start_idx + j + 1
                    obs = rec['text_obs']
                    action_str = rec['action']
                    # Convert action dict string to DSL format
                    action_dsl = self._format_action_as_dsl(action_str)
                    lines.append(f"Observation {step_num}: {obs}, Action {step_num}: {action_dsl}.\n")
                
                formatted_memory_contexts.append("\n".join(lines))
            
            memory_contexts = formatted_memory_contexts
            
        for i in range(len(text_obs)):
            admissible_actions_list = self.envs.get_admissible_commands[i]
            # Join as plain text; content already includes format and <action> examples where needed
            reformatted_admissible_actions = "\n".join(admissible_actions_list)
            admissible_actions_by_env.append(reformatted_admissible_actions)
            
            # Format goal for display (excludes internal normalization parameters)
            formatted_goal = self.format_goal_for_display(self.goals[i])
            
            
            prompt_goal_style = self.goals[i].get('prompt_goal_style')
            optimization_hint = self.goals[i].get('optimization_hint', '')
            
            if init or self.config.env.history_length <= 0:
                # Initial observation or no history
                template = self._circuitop_template(self.goals[i], with_history=False)
                obs = template.format(
                    goal=formatted_goal,
                    optimization_hint=optimization_hint,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # With history
                template = self._circuitop_template(self.goals[i], with_history=True)
                obs = template.format(
                    goal=formatted_goal,
                    optimization_hint=optimization_hint,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        Process batch for success metrics.
        Keep only global success metrics (no topology-wise delta/performance logging).
        """
        active_indices = [
            i for i, batch_item in enumerate(total_batch_list[batch_idx]) if batch_item['active_masks']
        ]
        if not active_indices:
            return

        # Keep final-step semantics for success/performance.
        final_idx = active_indices[-1]
        info = total_infos[batch_idx][final_idx]

        # Track different success metrics:
        # 1. success: gain+bandwidth+power with PM gate (benchmark core success)
        # 2. performance_success: all active hard constraints satisfied
        # 3. performance_success_any_step: any-step all-constraint success
        try:
            from agent_system.environments.env_package.circuitop.circuitop.rewards import check_core_success
            core_success = float(check_core_success(info, info.get('goal', {}) or {}))
        except Exception:
            core_success = float(
                info.get(
                    'success',
                    info.get(
                        'core_success',
                        info.get('performance_success', 0),
                    ),
                )
            )
        performance_success = float(info.get('performance_success', 0))

        performance_success_any_step = 0.0
        for idx in active_indices:
            step_info = total_infos[batch_idx][idx]
            if not isinstance(step_info, dict):
                continue
            if float(step_info.get('performance_success', 0)) > 0.0:
                performance_success_any_step = 1.0
                break

        # Main metrics
        success['success_rate'].append(core_success)                        # Core success rate (gain+bw+power+PM)
        success['performance_success_rate'].append(performance_success)     # All-constraint success rate
        success['performance_success_rate_any_step'].append(performance_success_any_step)
        return
    
    def success_evaluator(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Override success evaluator for CircuitOp and return global success metrics only.
        """
        from collections import defaultdict
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        sampled_topologies = set()
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
            active_indices = [
                i for i, batch_item in enumerate(total_batch_list[bs]) if batch_item['active_masks']
            ]
            if active_indices:
                final_idx = active_indices[-1]
                info = total_infos[bs][final_idx]
                if isinstance(info, dict):
                    topo = info.get('data_source') or info.get('topology')
                    if topo:
                        sampled_topologies.add(str(topo))
        
        assert len(success['success_rate']) == batch_size
        result = {key: np.array(value) for key, value in success.items()}
        # topologies drawn this batch, logged by the trainer
        result['sampled_topologies'] = sampled_topologies
        return result

def circuitop_env_kwargs(config):
    """The settings every CircuitopWorker is built with (training, evaluation and the demo)."""
    c = config.env.circuitop
    return {
        'workdir': c.workdir,
        'max_steps': config.env.max_steps,
        'circuit_name': c.get('circuit_name', 'default_circuit'),
        'representation': c.get('representation', 'xml'),
        'reward_config': c.get('reward_config', None),   # <yaml>:<preset>, package-relative
        'action_change_mode': c.get('action_change_mode'),   # percentage | absolute; None: the preset's
        'tech_node': c.get('tech_node'),                 # CIRCUITOP_TECH_NODE of the worker
        'tech_model': c.get('tech_model'),
    }


def make_circuitop_projection(config):
    """The action projection with the run's format settings (env.circuitop.*), as training, evaluation and the demo
    (`slic solve`, the browser UI) use it:
      enforce_think_prefix     the response must start with <think>
      reject_trailing_content  nothing may follow the final </action>
      require_think_tags       a <think></think> block is required (False relaxes the gate for models without one)
      think_prefilled          the chat template ends the generation prompt with "<think>"; the projection restores the tag"""
    from agent_system.environments.env_package.circuitop import circuitop_projection
    c = config.env.circuitop
    return partial(
        circuitop_projection,
        enforce_think_prefix=bool(c.get('enforce_think_prefix', False)),
        reject_trailing_content=bool(c.get('reject_trailing_content', False)),
        require_think_tags=bool(c.get('require_think_tags', True)),
        think_prefilled=bool(c.get('think_prefilled', False)),
    )


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    # For validation, use val_kwargs.n if available, otherwise use 1
    val_group_n = config.actor_rollout_ref.rollout.val_kwargs.get('n', 1) if config.actor_rollout_ref.rollout.val_kwargs.get('n', 1) > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "circuitop" in config.env.env_name.lower():
        from agent_system.environments.env_package.circuitop import build_circuitop_envs, circuitop_projection
        
        env_kwargs = circuitop_env_kwargs(config)   # task_data / target / topology come per row through reset()
        
        _envs = build_circuitop_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            resources_per_worker=resources_per_worker,
            is_train=True,
            env_kwargs=env_kwargs,
        )
        _val_envs = build_circuitop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=val_group_n,
            resources_per_worker=resources_per_worker,
            is_train=False,
            env_kwargs=env_kwargs,
        )
        
        projection_f = make_circuitop_projection(config)
        envs = CircuitopEnvironmentManager(_envs, projection_f, config)
        val_envs = CircuitopEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
