# CircuitOp prompt templates

Loaded by `agent_system/environments/prompts/circuitop.py`; the reward preset's `goal_prompt_style` picks the pair.

| file | `goal_prompt_style` | used for |
|---|---|---|
| `student_goal_with_optimization_hint_no_his.md` | `goal_with_optimization_hint` (default) | first turn |
| `student_goal_with_optimization_hint.md` | `goal_with_optimization_hint` (default) | later turns (adds recent observations and actions) |
| `student_goal_no_hint_no_his.md` | `goal_no_hint` (hint ablation) | first turn |
| `student_goal_no_hint.md` | `goal_no_hint` (hint ablation) | later turns |

The no-hint prompt is the hint prompt without the `{optimization_hint}` slot and without the Priority Rule / Validity Recovery
bullets. Placeholders (`{goal}`, `{optimization_hint}`, `{current_observation}`, `{admissible_actions}`, and for later turns
`{step_count}`, `{history_length}`, `{action_history}`, `{current_step}`) are filled with Python `str.format`; any other
placeholder is rejected when the module is imported.
