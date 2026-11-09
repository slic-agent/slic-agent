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

import os
import sys

import gymnasium as gym
import numpy as np
import ray
import torch
import torchvision.transforms as T
import yaml
from gymnasium import spaces

from agent_system.environments.env_package.circuitop.circuitop import CircuitOpEnv
from agent_system.environments.env_package.circuitop.circuitop.representation import resolve_representation
from agent_system.environments.env_package.circuitop.reward_config import (
    RewardConfig,
    RewardComputer,
    load_reward_config
)

def compute_reward(info, multi_modal=False, reward_computer=None):
    """Reward from reward_computer, or a sparse 10 * performance_success when none is given."""
    if reward_computer is None:
        # Fallback to simple sparse reward if no reward_computer configured
        performance_success = bool(info.get('performance_success', False))
        reward = 10.0 * float(performance_success)
        return reward
    

    # Use RewardComputer to get final reward
    final_reward = reward_computer.compute_reward(info)
    return final_reward

class CircuitopWorker:
    """Ray actor holding one CircuitOpEnv instance."""
    
    def __init__(self, env_kwargs):
        """Build the env from static settings (circuitop_env_kwargs); dataset fields arrive per episode in reset()."""
        # Extract reward config for reward computation settings
        reward_config_dict = env_kwargs.get('reward_config', {})
        self.reward_config = load_reward_config(reward_config_dict)
        mode = env_kwargs.get('action_change_mode')   # env.circuitop.action_change_mode overrides the preset's
        if mode:
            if mode not in ('percentage', 'absolute'):
                raise ValueError(f"env.circuitop.action_change_mode must be percentage or absolute, got {mode!r}")
            self.reward_config.action_change_mode = mode

        # Tech node / model file: environment variables local to this worker process.
        tech_node = env_kwargs.get('tech_node')
        tech_model = env_kwargs.get('tech_model')
        if tech_model:
            os.environ['CIRCUITOP_TECH_MODEL'] = str(tech_model)
        elif tech_node:
            os.environ['CIRCUITOP_TECH_NODE'] = str(tech_node)
        
        # Default goal from reward_config; reset() rebuilds it with the row's goal.
        self.reward_computer = RewardComputer(self.reward_config)
        
        # Store configuration for later use
        # env.circuitop.representation (None = not set); the row's value is combined with it at reset (circuitop/representation.py)
        self.configured_representation = env_kwargs.get('representation')
        self.representation = None
        self.task_data = None  # Will be set during reset()
        self.initial_obs_xml = None  # Will be set during reset()
        
        # Use topology from dataset for data_source (per-topology metrics)
        # Falls back to circuit_name from config if topology not in dataset
        topology = env_kwargs.get('topology', env_kwargs.get('circuit_name', 'default_circuit'))
        
        self.env = CircuitOpEnv(
            workdir=env_kwargs['workdir'],
            max_steps=env_kwargs.get('max_steps', 50),
            circuit_name=topology,  # Use topology for data_source field
            goal=self.reward_config.goal,  # Default goal from config (updated in reset)
            reward_config=self.reward_config,  # Pass updated config
        )

    def get_state(self):
        """Snapshot worker+env episode state (SLIC continuation sync)."""
        import copy
        worker_state = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "env"}
        return {"env": self.env.get_state(), "worker": worker_state}

    def set_state(self, state):
        import copy
        for k, v in state["worker"].items():
            setattr(self, k, copy.deepcopy(v))
        self.env.set_state(state["env"])
        return True

    def _augment_goal_prompt_fields(self, target: dict) -> dict:
        """Prompt fields of the goal for the preset's goal_prompt_style (the two released prompts)."""
        target = dict(target or {})
        style = getattr(self.reward_config, "goal_prompt_style", None)
        if style not in ("goal_with_optimization_hint", "goal_no_hint"):
            raise ValueError(f"goal_prompt_style must be goal_with_optimization_hint or goal_no_hint, got {style!r}.")
        target["prompt_goal_style"] = style

        # goal_no_hint: explicit goal specs, no optimisation / recovery guidance anywhere in the prompt.
        if style == "goal_no_hint":
            target["optimization_hint"] = ""
            return target
        hint = getattr(self.reward_config, "goal_optimization_hint", None)
        if not hint:
            raise ValueError("goal_prompt_style goal_with_optimization_hint needs the preset's goal_optimization_hint text.")
        target["optimization_hint"] = hint
        return target
    
    def step(self, action):
        """Execute a step and return reward from environment."""
        obs, reward, done, info = self.env.step(action)
        
        # Reward already computed by the env; wrap values in lists for CircuitopEnvs.step().
        obs = [obs]
        reward = [reward]
        done = [done]
        info = {k: [v] for k, v in info.items()}
        
        return obs, reward, done, info
    
    def reset(self, task_data=None, target=None, initial_obs_xml=None, **kwargs):
        """
        Reset environment with dataset fields.
        
        Args:
            task_data: Circuit structure and parameters from dataset
            target: Goal dict from dataset (overrides reward_config.goal)
            initial_obs_xml: Initial observation XML from dataset
        """
        # Representation: env.circuitop.representation when set, else the row's env_kwargs.representation, else an error
        # (circuitop/representation.py; the environment manager's system prompt uses the same rule).
        self.representation = resolve_representation(self.configured_representation, kwargs.get("representation"), task_data,
                                                     task_id=str((task_data or {}).get("metadata", {}).get("task_id", "")) if isinstance(task_data, dict) else "")
        
        task_metadata = {}
        if task_data is not None and isinstance(task_data, dict):
            task_metadata = task_data.get('metadata', {}) or {}
        
        # Row goal from the dataset
        if target is not None:
            # Default bandwidth goal 10 MHz if absent; the phase-margin window is always 30-90 deg.
            if 'bandwidth_min' not in target:
                target = {**target, 'bandwidth_min': 10.0}
            target = {
                **target,
                'phase_margin_min_deg': 30.0,
                'phase_margin_max_deg': 90.0,
            }
            target = self._augment_goal_prompt_fields(target)

            self.reward_config.goal = target
            self.reward_computer = RewardComputer(self.reward_config)
            
            # Enrich the raw row goal with the preset's tolerances and settings; the preset's
            # constraint_specs (success, stop rule and per-turn score) override any list in the row.
            constraint_specs = list(self.reward_config.constraint_specs)

            enriched_fields = {
                'allow_further_exploration_after_goal': self.reward_config.allow_further_exploration_after_goal,
                'gain_improvement_tolerance_dB': self.reward_config.gain_improvement_tolerance_dB,
                'gain_relative_tolerance': self.reward_config.gain_relative_tolerance,
                'terminal_bonus_gain_relative_tolerance': self.reward_config.terminal_bonus_gain_relative_tolerance,
                'power_relative_tolerance': self.reward_config.power_relative_tolerance,
                'bandwidth_relative_tolerance': self.reward_config.bandwidth_relative_tolerance,
                'phase_margin_relative_tolerance': self.reward_config.phase_margin_relative_tolerance,
                'action_change_mode': self.reward_config.action_change_mode,
                'goal_prompt_style': self.reward_config.goal_prompt_style,
                'topology_complexity': task_metadata.get('complexity'),
                'signal_mode': task_metadata.get('signal_mode'),
            }
            if self.reward_config.goal_prompt_style == 'goal_no_hint':
                if self.reward_config.goal_optimization_hint:
                    raise ValueError(
                        "goal_prompt_style=goal_no_hint forbids goal_optimization_hint "
                        f"(got {self.reward_config.goal_optimization_hint!r})."
                    )
                enriched_fields['optimization_hint'] = ''
            elif self.reward_config.goal_optimization_hint:
                enriched_fields['optimization_hint'] = self.reward_config.goal_optimization_hint
            if constraint_specs is not None:
                enriched_fields['constraint_specs'] = constraint_specs

            
            # Update environment's goal with enriched version
            self.env.goal = {**target, **enriched_fields}
        
        # Pass dataset fields to environment reset
        obs, info = self.env.reset(
            task_data=task_data,
            initial_obs_xml=initial_obs_xml,
            representation=self.representation,
            topology=kwargs.get("topology"),   # the row's topology; this worker instance serves rows of different topologies
        )
        
        obs = [obs]  # Wrap in list for compatibility
        
        # Enriched goal (env.goal, not the raw reward_config.goal) for rewards and prompt selection
        info['goal'] = self.env.goal
        
        info = {k: [v] for k, v in info.items()}
        return obs, info


class CircuitopEnvs(gym.Env):
    def __init__(self, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
        super().__init__()
        
        # Standalone runs only (the trainer inits Ray first): set CIRCUITOP_RAY_NUM_CPUS to the CPU allocation
        # (bare ray.init() prestarts a worker per physical core) and a per-job RAY_TMPDIR when jobs share a node
        # (a shared /tmp/ray session deadlocks).
        if not ray.is_initialized():
            _num_cpus = os.environ.get("CIRCUITOP_RAY_NUM_CPUS")
            _tmp_dir = os.environ.get("RAY_TMPDIR") or None
            if _num_cpus:
                ray.init(num_cpus=int(_num_cpus), _temp_dir=_tmp_dir)
            else:
                ray.init(_temp_dir=_tmp_dir)
        
        if env_kwargs is None:
            env_kwargs = {}
            
        self.multi_modal = False  # Set to True if your env uses images
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.is_train = is_train
        np.random.seed(seed)

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(CircuitopWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(env_kwargs)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            # Reward already computed by the worker
            rewards_list.append(scores[0])

        return text_obs_list,rewards_list, dones_list, info_list

    def reset(self, kwargs=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        
        Args:
            kwargs: List of dicts with dataset fields for each worker (task_data, target, initial_obs_xml)
        """
        if kwargs is None:
            kwargs = [{}] * len(self.workers)
        
        # Pad with the last row's kwargs when batch_size < num_workers (e.g. last eval batch)
        if len(kwargs) > 0 and len(kwargs) < len(self.workers):
            last_kwargs = kwargs[-1]  # Reuse last sample's kwargs for extra workers
            kwargs = list(kwargs) + [last_kwargs] * (len(self.workers) - len(kwargs))
        
        text_obs_list = []
        info_list = []

        # Send reset commands to all workers with their kwargs
        futures = []
        for i, worker in enumerate(self.workers):
            worker_kwargs = kwargs[i] if i < len(kwargs) else {}
            future = worker.reset.remote(**worker_kwargs)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        return text_obs_list, info_list


    def get_states(self, indices):
        """Fetch episode-state snapshots for the given worker indices."""
        return ray.get([self.workers[int(i)].get_state.remote() for i in indices])

    def set_states(self, assignments):
        """assignments: list of (worker_index, state) to restore."""
        ray.get([self.workers[int(i)].set_state.remote(s) for i, s in assignments])

    @property
    def get_admissible_commands(self):
        """Admissible commands per env, as recorded at the last step/reset."""
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_circuitop_envs(seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
    """Build env_num groups of group_n CircuitopWorker actors, each from env_kwargs (see circuitop_env_kwargs)."""
    return CircuitopEnvs(seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
