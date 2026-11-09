"""Fields written to rollout dumps (one JSON object per generated turn).

Shared by the three dump writers: training rollouts (``trainer.rollout_data_dir``, including pre-filter dumps),
training-time validation (``trainer.validation_data_dir``) and offline evaluation (``verl/trainer/main_env_eval.py``).
A writer keeps only the listed fields it has, in this order; fields it does not produce are absent (e.g. advantages
exist only in training dumps). Unlisted step info (duplicate aliases, nested dicts, run-constant config, timers) is
dropped. Per-trajectory sums other than ``episode_reward`` are not stored: sum ``score`` over a ``traj_uid``.
"""

DUMP_FIELDS = (
    # --- which task / episode / turn this row is
    "step",                      # training: global step of the dump
    "batch",                     # evaluation: batch index
    "task_id", "topology", "topology_complexity", "goal_level",
    "uid",                       # group id: rows sharing a state and turn (training)
    "traj_uid",                  # one id per trajectory
    "prompt_group_id", "prompt_sample_index",   # evaluation: task and sample index among its n samples
    "turn_index",                # 0-based turn of this row
    "is_final_step", "is_padding_sample",
    # --- the turn
    "input", "output", "turn_response_length",
    # --- outcome of the action
    "is_action_valid", "error", "error_type",
    "gain_dB", "power_uW", "bandwidth_MHz", "bandwidth_kind", "inband_flatness_db", "phase_margin_deg", "phase_margin_status",
    "offset_v", "cmrr_db", "psrr_plus_db", "psrr_minus_db",
    "initial_gain_dB", "initial_power_uW",
    "resulting_params",          # device parameters after the action
    # --- the four targets and the per-spec verdicts
    "goal_gain_min", "goal_power_max", "goal_bandwidth_min", "goal_phase_margin_min_deg", "goal_phase_margin_max_deg",
    "gain_pass", "power_pass", "bandwidth_pass", "phase_margin_pass",
    "performance_success", "done",
    # --- rewards
    "score",                     # training: the per-turn reward the estimator trained on (turn-level estimators: the
                                 # step_wise reward of the post-action state, the fixed reward for an invalid turn);
                                 # evaluation: the environment's reward
    "env_reward",                # training: the environment's own reward (0 before the done turn, the terminal metric bonus at it)
    "perf_score_pre", "perf_score_post",   # four-spec score of the state before / after the action
    "episode_reward",            # the environment's episode total, on every row of the trajectory
    "episode_length",
    # --- advantages (training)
    "step_wise_validity_advantage", "step_wise_performance_advantage", "step_wise_performance_eligible",
    "step_wise_advantage",       # the row's advantage (constant over the response tokens of a turn)
    "step_wise_signal_kept",     # dynamic filtering (pre-filter dumps): the row's group was kept
    "slic_continuation_selected",   # 1 on the branch replayed as the next turn's shared state, else 0
    # MT-GRPO tree: node ids, depth (= turn_index + 1), terminal flag, the outcome R^O the estimator used (NaN on
    # non-terminal rows) and the node advantage
    "mtg_node_id", "mtg_parent_id", "mtg_depth", "mtg_is_terminal", "mtg_outcome_reward", "mtg_advantage",
    # --- timing (evaluation)
    "episode_wall_clock_sec",
)


def tidy_dump_row(entry: dict) -> dict:
    """Keep only DUMP_FIELDS, in their order. The training writer stores the turn as ``turn_idx`` when the batch carries no
    ``turn_index``; it is written as ``turn_index`` (the same 0-based integer)."""
    if "turn_index" not in entry and "turn_idx" in entry:
        entry = {**entry, "turn_index": entry["turn_idx"]}
    return {k: entry[k] for k in DUMP_FIELDS if k in entry}
