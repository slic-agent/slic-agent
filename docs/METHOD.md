# Method notes

## Task

An episode starts from a circuit topology with an initial design and a goal of four specifications: minimum gain,
maximum power, minimum bandwidth and a phase-margin window. Each turn the policy reads the circuit and its measured
metrics, reasons inside `<think>...</think>` and emits an `<action>` that changes device parameters or declares the task
done; the environment simulates the new design and returns its metrics. An episode succeeds when all four specifications
are met within the preset tolerances (four-spec success). Evaluation runs 8 turns, training 4 (`scripts/train/run_slic.sh`; the
paper trained with 8); the prompt carries the previous turn's observation (`history_length: 1`).

Two switches change what the model sees and writes (the paper used the defaults):

- `env.circuitop.representation`: `xml` (default; the circuit as a graph, devices named by handle, `@10`) or `netlist` (the
  annotated netlist, devices named by id, `X0.MM0`);
- `env.circuitop.action_change_mode`: `percentage` (default; `<action>increase @10 w_MM30 15%</action>`) or `absolute`
  (`<action>w_MM30=420n; iref=10u</action>`).

Power is the DC supply power of one amplifier and its bias block. A complex topology's test bench simulates five copies
of the circuit (differential gain, common-mode gain, DC operating point, PSRR+, PSRR−), each on its own supply pair, and
counts only the differential-gain copy (the sources named by the netlist's `CIRCUITOP_POWER_SOURCES` marker).

## SLIC rollout tree

For every task, G branches are sampled at turn 1. After every turn SLIC picks **one** branch and replays it as the shared
state of all G samples of the next turn, so the G branches of a turn share their history and their rewards are directly
comparable; the stepwise advantage is computed within each turn's group. The branch is chosen by
`algorithm.slic.continuation_policy`:

| policy       | selection rule                                              |
|--------------|-------------------------------------------------------------|
| `random`     | uniform over the eligible branches (**SLIC-Random**, main)  |
| `softmax`    | probability ∝ exp(score / T)                                |
| `greedy`     | the branch with the best continuation score                 |
| `fixed_base` | no online continuation: all turns branch from one base trajectory (the fixed-base baseline) |

The score is the post-action performance score; `continuation_valid_only` restricts the choice to simulatable branches.

## Advantages

`algorithm.step_wise.advantage_mode`:

- `separate_validity_perf` (main): a validity channel (the action simulated and kept the circuit in a valid operating
  region) and a performance channel (specification progress), normalised separately within the group and summed with
  `validity_weight` / `performance_weight`;
- `joint`: one scalar reward per turn, normalised once.

`skip_degenerate_groups` (dynamic filtering) drops groups whose rewards are all equal.

The MT-GRPO baseline (`algorithm.adv_estimator=mt_grpo_tree`, `configs/train/mt_grpo.yaml`) follows MT-GRPO's per-state
rollouts (arXiv 2505.11821): every state at turns 1..K−1 is sampled G = `algorithm.mt_grpo.branching` times and turn
K takes one continuation, giving G^(K−1) trajectories per task (`env.rollout.n` must equal it). A^I is the z-score of the
turn reward among the G siblings, A^O that of the outcome over the task's trajectories; a node's advantage is
A^I + α · mean(children), a terminal node takes A^O, and each generated turn is trained once. The turn reward is SLIC's
per-turn score, the outcome the same score at the final state.

## Rewards

The presets are in `agent_system/environments/env_package/circuitop/configs/reward_config_example.yaml`, each with a
comment naming the configs that use it. Name parts: `sparse_constraint_done_4spec` (four-spec goal), `hint` / `nohint`
(the prompt does or does not carry the optimisation hint, which asks to recover gain first), `tolalign` (tolerance-aligned
score), `contoptstop` (stop at success) / `contopt` (continue optimising after success).

**Per-turn score.** `perf_score_post`, the four-spec score of the state after the action (an unparseable or failed action
gets `step_wise.invalid_turn_reward`). Before success it is the tolerance-aligned mean of the four specification terms
(< 1); once all four pass it is `1 + 0.5 × mean(slack terms)`, so every success ranks above every failure:

- gain, bandwidth, power: the relative slack to the goal, shifted by the tolerance and capped at 1;
- phase margin (a window [PM_min, PM_max]): 1 − |PM − c| / h, with c the window centre and h the tolerance-widened
  half-width, clipped to [0, 1]; a design with no unity-gain crossing scores 1.

**Termination.** An episode ends at the turn limit or, with `contoptstop`, as soon as every specification passes. The
stepwise estimators (SLIC, the fixed-base baseline, MT-GRPO) train on the per-turn score; the episode-level GRPO reference trains on the
environment's episode reward (the terminal metric bonus at the done step, 0 before).

**Success metrics** (`results.json`, W&B). `success_rate`: all four specifications met at the final turn; the benchmark
verdict, used for checkpoint selection (`val/success_rate`) and reported in the paper. `performance_success_rate`: every
specification of the preset's `constraint_specs` met; it ends the episode and is the success the reward sees. All released
presets list the same four specifications, so the two coincide. The any-turn forms are
`performance_success_rate_any_step` (training / validation) and `any_turn_success_rate` (evaluation).

**Config layout.** `algorithm.slic` holds the continuation settings, `algorithm.step_wise` the per-turn reward, advantage
shaping and batch assembly; the two are merged at start-up (`verl/utils/slic_groups.py`), so a key appears in only one of
them. `algorithm.step_wise.target_train_rows` (1024) caps the training rows per step.
