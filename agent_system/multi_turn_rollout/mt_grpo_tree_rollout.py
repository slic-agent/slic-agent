"""MT-GRPO full-tree rollout for circuitop (arXiv 2505.11821 v3, "per-state (per-turn) rollouts"); training only.

Per task, one G-ary tree over K = env.max_steps turns: every non-terminal state at turns 1..K-1 is sampled G times
from the identical env state (zero-sim snapshot restore); at turn K each surviving state gets one continuation;
nodes whose episode ended are not expanded.

`env.rollout.n` must equal G^(K-1) slots per task. Each turn only slots holding a node are generated; idle slots
get an empty action, rejected by the projection before any simulation. Counts go to meta_info['mt_grpo_tree/<count>'].

Every node is emitted once as a training row, carrying uid (task), mtg_node_id, mtg_parent_id (sibling-group key),
mtg_depth (1-based turn), mtg_is_terminal and mtg_outcome_reward (terminal env reward; NaN otherwise).
The advantage is core_algos.compute_mt_grpo_tree_advantage (adv_estimator=mt_grpo_tree).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

import numpy as np

from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _cfg(collector, key: str, default=None):
    cfg = getattr(collector.config.algorithm, "mt_grpo", None)
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


class _Node:
    __slots__ = ("node_id", "parent", "depth", "task", "snapshot", "row", "info", "env_reward", "done", "children",
                 "traj_uid")

    def __init__(self, node_id, parent, depth, task, snapshot=None):
        self.node_id = node_id
        self.parent = parent
        self.depth = depth
        self.task = task
        self.snapshot = snapshot
        self.row = None
        self.info = None
        self.env_reward = 0.0
        self.done = False
        self.children: List["_Node"] = []
        self.traj_uid = None


def mt_grpo_tree_multi_turn_loop(collector, gen_batch, actor_rollout_wg, envs):
    """Breadth-first full-tree rollout. `gen_batch` arrives repeated by env.rollout.n (= slots per task).

    Returns the same 8-tuple as vanilla_multi_turn_loop, ready for gather_rollout_data."""
    G = int(_cfg(collector, "branching", 4))
    K = int(collector.config.env.max_steps)
    S = int(collector.config.env.rollout.n)
    if G < 2 or K < 1:
        raise ValueError(f"mt_grpo.branching={G}, env.max_steps={K}: need G >= 2, K >= 1")
    if S != G ** (K - 1):
        raise ValueError(f"mt_grpo tree needs env.rollout.n == G^(K-1) = {G ** (K - 1)} slots per task, got {S}")
    circuit_cfg = getattr(collector.config.env, "circuitop", None)
    batch_size = len(gen_batch)
    if batch_size % S:
        raise ValueError(f"batch {batch_size} not divisible by {S} slots per task")
    num_tasks = batch_size // S

    env_reset_kwargs = gen_batch.non_tensor_batch.get("env_kwargs", None)
    if "circuitop" in str(collector.config.env.env_name).lower() and env_reset_kwargs is None:
        raise ValueError("CircuitOp rollout requires non_tensor_batch['env_kwargs'].")
    obs, _ = envs.reset(kwargs=env_reset_kwargs)

    task_uids = [str(uuid.uuid4()) for _ in range(num_tasks)]
    root_snaps = envs.tree_capture_env_states([t * S for t in range(num_tasks)], obs)
    roots = [_Node(f"{task_uids[t]}/r", None, 0, t, snapshot=root_snaps[t]) for t in range(num_tasks)]
    frontier: List[_Node] = list(roots)
    counts = {"generated": 0, "idle_slot_steps": 0, "levels": 0, "per_level": []}
    episode_wall = 0.0

    for k in range(1, K + 1):
        if not frontier:
            break
        m = G if k < K else 1
        # slot assignment: task t's frontier nodes fill slots t*S .. t*S + len*m - 1, m consecutive slots per node
        assign: Dict[int, _Node] = {}
        per_task_used = [0] * num_tasks
        for node in frontier:
            t = node.task
            for _ in range(m):
                if per_task_used[t] >= S:
                    raise RuntimeError(f"task {t}: level {k} needs more than {S} slots (tree bookkeeping bug)")
                assign[t * S + per_task_used[t]] = node
                per_task_used[t] += 1
        active = np.array(sorted(assign), dtype=np.int64)
        obs = envs.tree_restore_env_states([(int(i), assign[int(i)].snapshot) for i in active], obs)

        t0 = time.perf_counter()
        sub_gen = gen_batch.select_idxs(active)
        sub_obs = collector._subset_obs(obs, active)
        batch = collector.preprocess_batch(gen_batch=sub_gen, obs=sub_obs)
        pop_tensor = ["input_ids", "attention_mask", "position_ids"]
        pop_non_tensor = [key for key in ("raw_prompt_ids", "multi_modal_data", "raw_prompt", "tools_kwargs")
                          if key in batch.non_tensor_batch]
        batch_input = batch.pop(batch_keys=pop_tensor, non_tensor_batch_keys=pop_non_tensor)
        batch_input.meta_info = gen_batch.meta_info
        padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        out = unpad_dataproto(actor_rollout_wg.generate_sequences(padded), pad_size=pad_size)
        batch = batch.union(out)
        texts = collector.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

        actions = [""] * batch_size          # idle slots: rejected by the projection before any simulation
        for j, i in enumerate(active):
            actions[int(i)] = texts[j]
        next_obs, rewards, dones, infos = envs.step(actions)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)
        if len(dones.shape) == 2:
            dones = dones.squeeze(1)
        rewards = torch_to_numpy(rewards)
        dones = np.asarray(torch_to_numpy(dones), dtype=bool)
        step_sec = time.perf_counter() - t0
        episode_wall += step_sec
        child_snaps = envs.tree_capture_env_states(active, next_obs)

        n_act = len(active)
        batch.non_tensor_batch["uid"] = np.array([task_uids[assign[int(i)].task] for i in active], dtype=object)
        batch.non_tensor_batch["is_action_valid"] = np.array(
            [int(infos[int(i)].get("is_action_valid", 1)) for i in active], dtype=np.int32)
        batch.non_tensor_batch["rollout_step_wall_clock_sec"] = np.full(n_act, step_sec, dtype=np.float32)
        batch.non_tensor_batch["rollout_episode_wall_clock_sec_so_far"] = np.full(n_act, episode_wall, dtype=np.float32)
        batch.non_tensor_batch["rewards"] = np.array([rewards[int(i)] for i in active], dtype=object)
        batch.non_tensor_batch["active_masks"] = np.ones(n_act, dtype=object)
        batch.non_tensor_batch["turn_index"] = np.full(n_act, k - 1, dtype=np.int32)
        for key in ("env_kwargs", "extra_info"):
            if key in sub_gen.non_tensor_batch and key not in batch.non_tensor_batch:
                batch.non_tensor_batch[key] = sub_gen.non_tensor_batch[key]
        rows = to_list_of_dict(batch)

        new_frontier: List[_Node] = []
        for j, i in enumerate(active):
            i = int(i)
            parent = assign[i]
            child = _Node(f"{parent.node_id}/{len(parent.children)}", parent, k, parent.task, snapshot=child_snaps[j])
            child.row = rows[j]
            info = infos[i] if isinstance(infos[i], dict) else {}
            info["rollout_step_index"] = int(k - 1)
            info["rollout_step_wall_clock_sec"] = float(step_sec)
            info["rollout_episode_wall_clock_sec_so_far"] = float(episode_wall)
            child.info = info
            child.env_reward = float(rewards[i])
            child.done = bool(dones[i]) or k == K
            parent.children.append(child)
            if not child.done:
                new_frontier.append(child)
        counts["generated"] += n_act
        counts["idle_slot_steps"] += batch_size - n_act
        counts["per_level"].append(n_act)
        counts["levels"] += 1
        frontier = new_frontier
        obs = next_obs

    # ---- assembly: one trajectory list per leaf; every node emitted once (in the first leaf's list that reaches it)
    out_batch: List[List[Dict[str, Any]]] = []
    out_infos: List[List[Dict[str, Any]]] = []
    full_infos: List[List[Dict[str, Any]]] = []
    full_rows: List[List[Dict[str, Any]]] = []
    out_rewards, out_lengths, out_wall, out_tool, out_traj = [], [], [], [], []

    def leaves(node):
        if not node.children:
            yield node
        for c in node.children:
            yield from leaves(c)

    for root in roots:
        for leaf in leaves(root):
            if leaf is root:
                raise RuntimeError("mt_grpo tree: a task produced no turns")
            path = []
            n = leaf
            while n is not root:
                path.append(n)
                n = n.parent
            path.reverse()
            traj = str(uuid.uuid4())
            rows_once, infos_once = [], []
            for node in path:
                if node.traj_uid is None:
                    node.traj_uid = traj
                    r = node.row
                    r["traj_uid"] = traj
                    r["mtg_node_id"] = node.node_id
                    r["mtg_parent_id"] = node.parent.node_id
                    r["mtg_depth"] = np.int32(node.depth)
                    r["mtg_is_terminal"] = bool(node.done)
                    r["mtg_outcome_reward"] = float(node.env_reward) if node.done else float("nan")
                    rows_once.append(r)
                    infos_once.append(node.info)
            out_batch.append(rows_once)
            out_infos.append(infos_once)
            full_rows.append([nd.row for nd in path])
            full_infos.append([nd.info for nd in path])
            out_rewards.append(float(sum(nd.env_reward for nd in path)))
            out_lengths.append(float(len(path)))
            out_wall.append(float(episode_wall))
            out_tool.append(0.0)
            out_traj.append(traj)

    # success over full root->leaf paths (every trajectory counted once, as in the vanilla loop)
    success = envs.success_evaluator(
        total_infos=full_infos, total_batch_list=full_rows,
        episode_rewards=np.asarray(out_rewards, dtype=np.float32),
        episode_lengths=np.asarray(out_lengths, dtype=np.float32),
    )
    n_rows = sum(len(r) for r in out_batch)
    counts.update(rows=n_rows, trajectories=len(out_batch), tasks=num_tasks)
    collector._mt_grpo_tree_last_counts = counts
    return (out_batch,
            np.asarray(out_rewards, dtype=np.float32),
            np.asarray(out_lengths, dtype=np.float32),
            np.asarray(out_wall, dtype=np.float32),
            success,
            np.asarray(out_traj, dtype=object),
            np.asarray(out_tool, dtype=np.float32),
            out_infos)
