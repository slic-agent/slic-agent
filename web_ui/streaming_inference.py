"""One CircuitOp episode, streamed turn by turn: the demo behind `slic solve` and the browser UI (`slic serve`).

The episode runs through the evaluation pipeline, so the model receives exactly the prompts of `slic eval`: the config is
env_evaluation.yaml with scripts/eval/run_eval.sh's environment / response-length / sampling overrides, then the caller's; one
in-process CircuitopWorker behind CircuitopEnvironmentManager; messages [system: CIRCUITOP_SYSTEM_HEADER + the circuit
structure, user: the manager's observation], as in agent_system/multi_turn_rollout/rollout_loop.py.
Models: provider "local" is any OpenAI-compatible server (e.g. `vllm serve <checkpoint>`), sampled with the evaluation
protocol unless overridden, and streamed; other providers go through circuit/agents/api_agent.Agent (not streamed).
Events, one JSON object per line: init (goal, sampling, simulated initial state and parameters), token (a piece of the
current reply, thinking included), step (one per turn, with the exact messages and reply; "error" when the model call
failed, "env_error" the environment's feedback on an invalid action), done, error. "step" is the 0-based turn index.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent_system.environments.prompts.circuitop import CIRCUITOP_SYSTEM_HEADER

EVAL_LAUNCHER = REPO / "scripts" / "eval" / "run_eval.sh"
CONFIG_DIR = REPO / "verl" / "trainer" / "config"
REWARD_CONFIG = "configs/reward_config_example.yaml"   # package-relative, as in the launchers
_PROTOCOL_KEYS = ("env.", "data.max_response_length=", "rollout.temperature=", "rollout.top_p=")


def protocol_overrides() -> List[str]:
    """The hydra overrides of scripts/eval/run_eval.sh's command line that shape an episode (env.*, the response length and the
    sampling): the launcher is the single source of the evaluation protocol."""
    tokens: List[str] = []
    for line in EVAL_LAUNCHER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens += shlex.split(line.rstrip("\\").strip())
    return [t for t in tokens if t.startswith(_PROTOCOL_KEYS)]


def protocol_sampling() -> Dict[str, Any]:
    """The evaluation protocol's sampling (scripts/eval/run_eval.sh): what a local model gets unless the caller overrides it."""
    cfg = episode_config()
    return {"temperature": float(cfg.rollout.temperature), "top_p": float(cfg.rollout.top_p),
            "max_tokens": int(cfg.data.max_response_length), "seed": None}


API_SAMPLING_DEFAULTS = {"temperature": 0.4, "top_p": None, "max_tokens": None, "seed": None}   # circuit/agents/api_agent.py


def episode_config(overrides: Sequence[str] = ()):
    """env_evaluation.yaml + the evaluation protocol + `overrides` (later ones win)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="env_evaluation", overrides=protocol_overrides() + list(overrides))


class _InProcessEnvs:
    """CircuitopEnvs (agent_system/environments/env_package/circuitop/envs.py) for one environment: the same CircuitopWorker,
    called directly instead of through a Ray actor, and the same unwrapping of its one-element lists."""

    def __init__(self, worker):
        self.worker = worker
        self.num_processes = 1
        self.prev_admissible_commands = [None]

    def reset(self, kwargs=None):
        obs, info = self.worker.reset(**(kwargs[0] if kwargs else {}))
        info = {k: v[0] for k, v in info.items()}
        self.prev_admissible_commands[0] = info["admissible_commands"]
        return [obs[0]], [info]

    def step(self, actions):
        obs, scores, dones, info = self.worker.step(actions[0])
        info = {k: v[0] for k, v in info.items()}
        self.prev_admissible_commands[0] = info["admissible_commands"]
        return [obs[0]], [scores[0]], [dones[0]], [info]

    @property
    def get_admissible_commands(self):
        return self.prev_admissible_commands


def _messages(obs: Dict[str, Any]) -> List[Dict[str, str]]:
    """The chat messages of one turn, exactly as rollout_loop.py builds them."""
    messages = []
    system = (obs.get("system") or [None])[0]
    if system:
        messages.append({"role": "system", "content": f"{CIRCUITOP_SYSTEM_HEADER}{system}"})
    messages.append({"role": "user", "content": obs["text"][0] or ""})
    return messages


def _reasoning_and_action(raw: str, think_prefilled: bool) -> Tuple[str, str]:
    text = raw if (not think_prefilled or raw.lstrip().lower().startswith("<think>")) else "<think>" + raw
    think = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    acts = re.findall(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    return (think[0].strip() if think else ""), " | ".join(a.strip() for a in acts)


def _num(info: Dict[str, Any], key: str) -> float:
    v = info.get(key)
    return float(v) if v is not None else 0.0


def _metric(info: Dict[str, Any], key: str) -> Optional[float]:
    """A measured quantity: None stays None (the environment's "not measurable", e.g. no -3 dB point), never 0."""
    v = info.get(key)
    return float(v) if v is not None else None


def run_streaming_inference(
    task: Dict[str, Any],
    provider: str,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    reward_preset: Optional[str] = None,
    tech_node: Optional[str] = None,
    local_model_base_url: Optional[str] = None,
    local_model_name: Optional[str] = None,
    goal_override: Optional[Dict[str, Any]] = None,
    initial_params_override: Optional[Dict[str, Any]] = None,
    overrides: Sequence[str] = (),
) -> Generator[str, None, None]:
    """Run one episode of `task` (a task row's env_kwargs: topology, target, task_data) and yield its events.

    max_steps / reward_preset / tech_node override the evaluation protocol; goal_override replaces target specs;
    initial_params_override replaces starting device parameters; overrides are extra hydra overrides. temperature / top_p /
    max_tokens / seed override the sampling (None = the protocol's for a local model, the API agent's default otherwise);
    with a seed, a local vLLM server reproduces the episode."""
    from agent_system.environments.env_manager import (CircuitopEnvironmentManager, circuitop_env_kwargs,
                                                       make_circuitop_projection)
    from agent_system.environments.env_package.circuitop.envs import CircuitopWorker

    workdir = tempfile.mkdtemp(prefix="slic_solve_")
    try:
        ov = [f"env.circuitop.workdir={workdir}"]
        if max_steps is not None:
            ov.append(f"env.max_steps={int(max_steps)}")
        if reward_preset:
            ov.append(f"env.circuitop.reward_config={REWARD_CONFIG}:{reward_preset}")
        if tech_node:
            ov.append(f"env.circuitop.tech_node={tech_node}")
        try:
            cfg = episode_config(ov + list(overrides))
        except Exception as e:
            yield json.dumps({"event": "error", "message": f"episode config: {type(e).__name__}: {e}"})
            return

        task = copy.deepcopy(task)
        target = dict(task.get("target") or {})
        target.update({k: v for k, v in (goal_override or {}).items() if v is not None})
        task["target"] = target
        if initial_params_override:
            init = task["task_data"].setdefault("initial_state", {})
            params = init.get("params") or {}
            own = {k for k, v in params.items() if v is not None}   # parquet rows carry every topology's keys, None = not this one
            given = {k: str(v).strip() for k, v in initial_params_override.items() if v is not None and str(v).strip() != ""}
            unknown = sorted(k for k in given if k not in own)
            if unknown:
                yield json.dumps({"event": "error", "message": f"starting sizes: {unknown[:8]} are not parameters of {task['topology']}"})
                return
            init["params"] = {**params, **given}

        use_local = provider == "local"
        if use_local and not (local_model_base_url and local_model_name):
            yield json.dumps({"event": "error", "message": "provider local needs local_model_base_url and local_model_name"})
            return
        try:
            if use_local:
                from openai import OpenAI
                client = OpenAI(base_url=local_model_base_url.rstrip("/"), api_key=os.environ.get("LOCAL_MODEL_API_KEY") or "EMPTY")
                sampling = {"temperature": float(cfg.rollout.temperature if temperature is None else temperature),
                            "top_p": float(cfg.rollout.top_p if top_p is None else top_p),
                            "max_tokens": int(cfg.data.max_response_length if max_tokens is None else max_tokens)}
                if seed is not None:
                    sampling["seed"] = int(seed)
            else:
                from circuit.agents.api_agent import Agent
                agent = Agent(provider=provider, model_name=model_name,
                              sampling={"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed})
        except Exception as e:
            yield json.dumps({"event": "error", "message": f"model setup failed: {type(e).__name__}: {e}"})
            return

        worker = CircuitopWorker(circuitop_env_kwargs(cfg))
        manager = CircuitopEnvironmentManager(_InProcessEnvs(worker), make_circuitop_projection(cfg), cfg)
        try:
            obs, infos = manager.reset([task])
        except Exception as e:
            yield json.dumps({"event": "error", "message": f"environment reset failed: {type(e).__name__}: {e}"})
            return
        info = infos[0]
        think_prefilled = bool(cfg.env.circuitop.get("think_prefilled", False))
        yield json.dumps({
            "event": "init",
            "topology": task.get("topology"),
            "max_steps": int(cfg.env.max_steps),
            # the sampling this episode uses (seed None = fresh samples every run)
            "sampling": dict(sampling, seed=sampling.get("seed")) if use_local else {**API_SAMPLING_DEFAULTS, **agent.sampling},
            "goal_display": manager.format_goal_for_display(manager.goals[0]),
            "goal_specs": {k: v for k, v in target.items() if v is not None},
            "initial_gain_dB": _metric(info, "initial_gain_dB"),
            "initial_power_uW": _metric(info, "initial_power_uW"),
            "initial_bandwidth_MHz": _metric(info, "bandwidth_MHz"),
            "initial_phase_margin_deg": _metric(info, "phase_margin_deg"),
            # the starting device parameters (the topology's own keys; a parquet row also lists other topologies' as None)
            "params": {k: v for k, v in ((task["task_data"].get("initial_state") or {}).get("params") or {}).items() if v is not None},
        })

        steps: List[Dict[str, Any]] = []
        best, best_score, best_netlist = -1, float("-inf"), ""
        for step_idx in range(int(cfg.env.max_steps)):
            messages = _messages(obs)
            try:
                if use_local:
                    # streamed: every delta is a `token` event (thinking included); the reply is their concatenation
                    parts: List[str] = []
                    if think_prefilled:   # the chat template prefills <think>; show it so the live text reads as the model's
                        yield json.dumps({"event": "token", "step": step_idx, "text": "<think>\n"})
                    for chunk in client.chat.completions.create(model=local_model_name, messages=messages, n=1, stream=True,
                                                                **sampling):
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            parts.append(delta)
                            yield json.dumps({"event": "token", "step": step_idx, "text": delta})
                    raw = "".join(parts)
                else:
                    system = messages[0]["content"][len(CIRCUITOP_SYSTEM_HEADER):] if messages[0]["role"] == "system" else None
                    raw, display, _ = agent.get_action(messages[-1]["content"], system_message=system)   # Agent adds the header
                    yield json.dumps({"event": "token", "step": step_idx, "text": display or raw})   # API: the whole reply at once
            except Exception as e:
                yield json.dumps({"event": "step", "step": step_idx, "error": f"{type(e).__name__}: {e}", "valid": False})
                break
            obs, rewards, dones, infos = manager.step([raw])
            info = infos[0]
            reasoning, action = _reasoning_and_action(raw, think_prefilled)
            valid = bool(info.get("is_action_valid", 0))
            score = _num(info, "perf_score_post")
            if valid and score > best_score:
                best, best_score, best_netlist = step_idx, score, getattr(worker.env, "netlist_text", "") or ""
            success = bool(info.get("performance_success", False))   # four-spec success, the evaluation's verdict
            step = {
                "step": step_idx,
                "reasoning": reasoning,
                "action": action,
                "gain_dB": _metric(info, "gain_dB"),
                "power_uW": _metric(info, "power_uW"),
                "bandwidth_MHz": _metric(info, "bandwidth_MHz"),
                "phase_margin_deg": _metric(info, "phase_margin_deg"),
                "valid": valid,
                "env_error": info.get("error") or None,   # the environment's feedback on an invalid action
                "reward": float(rewards[0]),
                "done": bool(dones[0]),
                "success": success,
                "goal_met": success,
                "perf_score_post": score,
                "is_best": best == step_idx,
                "params": dict(info.get("resulting_params") or {}),   # device parameters after this turn's action
            }
            if step["env_error"] is None:
                step.pop("env_error")
            steps.append(step)
            yield json.dumps({"event": "step", **step, "best_step_idx": best, "messages": messages, "response": raw})
            if dones[0]:
                break

        last = steps[-1] if steps else {}
        yield json.dumps({
            "event": "done",
            "best_step_idx": best,
            "total_steps": len(steps),
            "success": any(s.get("success") for s in steps),
            "final_gain_dB": last.get("gain_dB"),
            "final_power_uW": last.get("power_uW"),
            "final_bandwidth_MHz": last.get("bandwidth_MHz"),
            "final_phase_margin_deg": last.get("phase_margin_deg"),
            "best_netlist": best_netlist or (getattr(worker.env, "netlist_text", "") or ""),
        })
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
