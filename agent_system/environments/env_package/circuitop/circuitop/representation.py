"""Which observation representation (xml | netlist) an episode uses.

One rule for the environment worker (observation, action space) and the environment manager (system prompt), so the two never
disagree:
  1. `env.circuitop.representation`, when set, is used for every row;
  2. otherwise the row's `env_kwargs.representation`;
  3. otherwise there is no representation and this is an error (no default).
The chosen representation must exist in the row's task data (`structure` and, when present, `admissible_actions`).
"""
from __future__ import annotations

from typing import Any, Optional

VALID = ("xml", "netlist")
_notified: set = set()


def _norm(value: Any, source: str) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    rep = str(value).strip().lower()
    if rep not in VALID:
        raise ValueError(f"Invalid representation {value!r} from {source}; expected one of {VALID}.")
    return rep


def resolve_representation(configured: Any, row: Any, task_data: Optional[dict] = None, task_id: str = "") -> str:
    """Return the representation for one row; raise ValueError when none is given or the row lacks it."""
    conf, rowv = _norm(configured, "env.circuitop.representation"), _norm(row, "the row's env_kwargs.representation")
    rep = conf or rowv
    if rep is None:
        raise ValueError("No representation for this row: set env.circuitop.representation (xml | netlist) "
                         "or give the row an env_kwargs.representation.")
    if conf and rowv and conf != rowv and (conf, rowv) not in _notified:
        _notified.add((conf, rowv))
        print(f"[circuitop] representation {conf} (env.circuitop.representation) is used for rows that carry {rowv}", flush=True)
    if isinstance(task_data, dict) and task_data:
        for key in ("structure", "admissible_actions"):
            block = task_data.get(key)
            if key == "structure" and not isinstance(block, dict):
                raise ValueError(f"Row {task_id or '?'}: task_data has no structure block.")
            if isinstance(block, dict) and rep not in block:
                raise ValueError(f"Row {task_id or '?'}: representation {rep!r} is not in task_data.{key} (has {sorted(block)}).")
    return rep
