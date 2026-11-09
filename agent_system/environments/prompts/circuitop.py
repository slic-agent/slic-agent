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

"""CircuitOp prompt templates.

Two prompts, selected by the reward preset's ``goal_prompt_style``:

* ``goal_with_optimization_hint`` (default, every training preset): the goal, the preset's optimisation hint and a reasoning
  policy with a priority rule and a validity-recovery rule;
* ``goal_no_hint`` (hint ablation): the same prompt without the hint slot and without the priority / validity-recovery
  bullets.

Each exists in two forms: ``*_NO_HIS`` for the first turn (no history yet) and the full form, which adds the most recent
observations and actions. The text lives in ``circuitop_md/*.md`` (Python ``str.format`` placeholders); a missing file or a
placeholder outside the allowed set is an error.
"""
from pathlib import Path
from string import Formatter

# The system message of every turn: this header + the task's circuit structure (XML) — training and evaluation rollouts
# (agent_system/multi_turn_rollout/rollout_loop.py) and the demo (web_ui/streaming_inference.py, circuit/agents/api_agent.py)
CIRCUITOP_SYSTEM_HEADER = "You are an expert analog circuit designer. Use the following circuit structure:\n\n"

_MD_DIR = Path(__file__).with_name("circuitop_md")
_FIRST_TURN_FIELDS = {"goal", "current_observation", "admissible_actions"}
_HISTORY_FIELDS = _FIRST_TURN_FIELDS | {"step_count", "history_length", "action_history", "current_step"}


def _load(md_filename: str, allowed_fields: set) -> str:
    text = (_MD_DIR / md_filename).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"prompt template {md_filename} is empty")
    fields = {name for _, name, _, _ in Formatter().parse(text) if name}
    if not fields <= allowed_fields:
        raise ValueError(f"prompt template {md_filename} uses unknown placeholders {sorted(fields - allowed_fields)}")
    return text.rstrip() + "\n"


STUDENT_GOAL_WITH_OPTIMIZATION_HINT_CIRCUITOP_TEMPLATE_NO_HIS = _load(
    "student_goal_with_optimization_hint_no_his.md", _FIRST_TURN_FIELDS | {"optimization_hint"})
STUDENT_GOAL_WITH_OPTIMIZATION_HINT_CIRCUITOP_TEMPLATE = _load(
    "student_goal_with_optimization_hint.md", _HISTORY_FIELDS | {"optimization_hint"})
STUDENT_GOAL_NO_HINT_CIRCUITOP_TEMPLATE_NO_HIS = _load("student_goal_no_hint_no_his.md", _FIRST_TURN_FIELDS)
STUDENT_GOAL_NO_HINT_CIRCUITOP_TEMPLATE = _load("student_goal_no_hint.md", _HISTORY_FIELDS)

# goal_prompt_style -> (first-turn template, template with history)
CIRCUITOP_TEMPLATES = {
    "goal_with_optimization_hint": (STUDENT_GOAL_WITH_OPTIMIZATION_HINT_CIRCUITOP_TEMPLATE_NO_HIS,
                                    STUDENT_GOAL_WITH_OPTIMIZATION_HINT_CIRCUITOP_TEMPLATE),
    "goal_no_hint": (STUDENT_GOAL_NO_HINT_CIRCUITOP_TEMPLATE_NO_HIS, STUDENT_GOAL_NO_HINT_CIRCUITOP_TEMPLATE),
}
