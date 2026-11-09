"""
NetlistActionTool: apply percentage parameter changes to netlist text (strings only, no file I/O).
"""
import re
import json
import uuid
import math
from typing import List, Dict, Tuple, Optional

from .action_validator import ActionValidator


def _num_equal(a, b, tol=1e-9):
    """True if a and b (numbers or SI strings such as "1.5u") are equal within tol; else string equality."""
    try:
        # Try to parse as numbers (handles SI units if needed)
        val_a = float(a) if not isinstance(a, str) or not any(c.isalpha() for c in a) else _parse_si_simple(a)
        val_b = float(b) if not isinstance(b, str) or not any(c.isalpha() for c in b) else _parse_si_simple(b)
        return abs(val_a - val_b) <= tol
    except (ValueError, TypeError):
        # Fallback to string comparison
        return str(a) == str(b)


def _parse_si_simple(s):
    """Simple SI unit parser for action tool.

    Handles both SI-suffixed tokens ("1.3u") and scientific notation
    ("1.3e-06", read with its exponent, not as the mantissa 1.3, so a
    percentage action never writes a meter-scale value into the netlist).
    """
    UNIT_MAP = {'': 1, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9,
                'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9}
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().lower()
    # Scientific notation / plain numbers first — 'e' would otherwise be
    # cut off by the SI-suffix match below.
    try:
        return float(t)
    except ValueError:
        pass
    m = re.fullmatch(r'([-+]?[\d.]+(?:e[-+]?\d+)?)(meg|[fpnumkg])', t)
    if not m:
        return float(t)  # fail fast on garbage
    val, unit = m.groups()
    return float(val) * UNIT_MAP[unit]


class NetlistActionTool:
    """Turns a percentage action {direction, target_id, param, percentage} into netlist edits."""
    
    _PARAM_LINE_RE  = re.compile(r'^\s*parameters\b(.*)$', re.IGNORECASE)
    _PARAM_PAIR_RE  = re.compile(r'([A-Za-z_]\w*)=([^\s]+)')
    _PARAM_KV       = re.compile(r"(\b[a-zA-Z_]\w*?)=([^\s]+)")
    
    def __init__(self, handle_to_id: Optional[Dict[str, str]] = None):
        """handle_to_id: @handle -> instance name (e.g. {'@22': 'X0.R0'})."""
        self.handle_to_id = handle_to_id or {}

    # ---------- Parsing ----------
    def parse_actions(self, action_dict: Dict, netlist_text: str) -> List[Dict]:
        """Normalize a percentage ({direction, target_id, param, percentage}) or absolute ({param, value}) action;
        netlist_text supplies the current value. Returns a one-element list."""
        if not isinstance(action_dict, dict):
            raise ValueError(f"Expected action dict, got {type(action_dict)}")

        if 'direction' in action_dict and 'percentage' in action_dict:
            return self._convert_projection_action(action_dict, netlist_text)

        if 'param' in action_dict and ('value' in action_dict or 'to_value' in action_dict):
            return self._convert_absolute_param_action(action_dict, netlist_text)

        raise ValueError(
            "Invalid action dict format. Expected percentage mode keys "
            "(direction,target_id,param,percentage) or absolute mode keys (param,value)."
        )

    # ---------- Apply ----------
    def apply_actions(self, netlist_text: str, actions: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Apply actions to a Spectre netlist (string).
        Handles macro parameters (parameters block) or direct instance edits.

        Returns: (new_netlist_text, checks)
        """
        lines = netlist_text.splitlines()
        checks = []

        param_line_idxs, params_map, line_pairs = self._parse_parameters_block(lines)
        pending_updates: Dict[str, str] = {}

        for a in actions:
            scope = a.get("apply_scope", "instance")
            key = a["param"]
            v_from = a.get("from_value")
            v_to = a["to_value"]
            action_id = a["action_id"]

            if scope == "parameters":
                macro_key = self._lookup_ci_key(params_map, key)
                applied = False
                note = None
                from_matched = None
                if macro_key is None:
                    note = f"Parameter '{key}' not found in parameters block"
                else:
                    pv = params_map.get(macro_key)
                    from_matched = _num_equal(pv, v_from) if (pv is not None and v_from is not None) else None
                    pending_updates[macro_key] = v_to
                    applied = True
                checks.append({
                    "action_id": action_id,
                    "target_id": "parameters",
                    "inst_local": "parameters",
                    "param": key,
                    "applied": applied,
                    "applied_at": "parameters" if applied else None,
                    "macro_key": macro_key,
                    "from_matched": bool(from_matched) if from_matched is not None else None,
                    "note": note,
                })
                continue

            inst_full = a["target_id"]        # e.g., X0.M6
            inst_local = inst_full.split(".")[-1]  # "M6"
            j = self._find_inst_line_idx(lines, inst_local)

            applied = False
            from_matched = None
            applied_at = None
            macro_key = None
            note = None

            if j is not None:
                pm = self._param_map(lines[j])
                resolved_key = self._resolve_param_on_instance_line(key, pm)
                if resolved_key is not None:
                    key = resolved_key
                cur_val = pm.get(key)
                # macro reference?
                macro_key_lookup = None
                if cur_val is not None and re.match(r'^[A-Za-z_]\w*$', cur_val):
                    macro_key_lookup = self._lookup_ci_key(params_map, cur_val)
                if cur_val is not None and macro_key_lookup is not None:
                    macro_key = macro_key_lookup
                    pv = params_map.get(macro_key)
                    from_matched = _num_equal(pv, v_from) if (pv is not None and v_from is not None) else None

                    pending_updates[macro_key] = v_to
                    applied = True
                    applied_at = "parameters"
                else:
                    if cur_val is None:
                        note = f"Parameter token '{a['param']}' did not match any key/value on instance {inst_local}"
                    else:
                        from_matched = _num_equal(cur_val, v_from) if v_from is not None else None
                        try:
                            lines[j] = self._set_param_in_line(lines[j], key, v_to)
                        except re.error as e:
                            raise RuntimeError(
                                f"_set_param_in_line failed: key={key!r} val={v_to!r} line={lines[j]!r}"
                            ) from e

                        applied = True
                        applied_at = "instance_line"
            else:
                note = f"Instance {inst_local} (from {inst_full}) not found"

            checks.append({
                "action_id": action_id,
                "target_id": inst_full,
                "inst_local": inst_local,
                "param": key,
                "applied": applied,
                "applied_at": applied_at,
                "macro_key": macro_key,
                "from_matched": bool(from_matched) if from_matched is not None else None,
                "note": note
            })

        if pending_updates and param_line_idxs:
            self._rewrite_parameters_lines(lines, param_line_idxs, line_pairs, pending_updates)

        return "\n".join(lines), checks

    # ---------- Helpers ----------
    def _param_map(self, line: str):
        return {k: v for k, v in self._PARAM_KV.findall(line)}

    def _lookup_ci_key(self, mapping: Dict[str, str], key: str) -> Optional[str]:
        """Return the actual mapping key via case-insensitive lookup."""
        if key in mapping:
            return key
        key_lc = str(key).lower()
        for k in mapping.keys():
            if str(k).lower() == key_lc:
                return k
        return None

    def _resolve_param_on_instance_line(self, param_token: str, param_map: Dict[str, str]) -> Optional[str]:
        """
        Resolve action parameter token against one instance parameter map.

        Supports direct keys (w/l/r/c/dc) and macro/shared names in values
        (e.g., token Mpair_w resolves to key w for `w=Mpair_w`).
        """
        for candidate in ActionValidator._extract_param_candidates(param_token):
            key_match = self._lookup_ci_key(param_map, candidate)
            if key_match is not None:
                return key_match
            tok_lc = str(candidate).lower()
            for k, v in param_map.items():
                if isinstance(v, str) and v.lower() == tok_lc:
                    return k
        return None

    def _find_inst_line_idx(self, lines: List[str], inst_id: str) -> Optional[int]:
        pat = re.compile(rf"^\s*{re.escape(inst_id)}\b", re.IGNORECASE)
        for i, l in enumerate(lines):
            if pat.match(l):
                return i
        return None

    def _set_param_in_line(self, line: str, key: str, val: str) -> str:
        """Set parameter value in a netlist line."""
        pattern = rf"({re.escape(key)}=)\S+"   # matches l=... or w=... etc.
        return re.sub(pattern, lambda m: m.group(1) + str(val), line)

    def _parse_parameters_block(self, lines: List[str]):
        param_line_idxs, params, line_pairs = [], {}, []
        for i, l in enumerate(lines):
            m = self._PARAM_LINE_RE.match(l)
            if not m:
                continue
            param_line_idxs.append(i)
            tail = m.group(1)
            pairs = self._PARAM_PAIR_RE.findall(tail)
            line_pairs.append(pairs)
            for k, v in pairs:
                params[k] = v
        return param_line_idxs, params, line_pairs

    def _rewrite_parameters_lines(
        self,
        lines: List[str],
        param_line_idxs: List[int],
        line_pairs: List[List[Tuple[str, str]]],
        updates: Dict[str, str]
    ) -> None:
        """Rewrite parameters block with updated values."""
        for idx, pairs in zip(param_line_idxs, line_pairs):
            new_pairs = []
            for k, v in pairs:
                new_pairs.append(f"{k}={updates.get(k, v)}")
            lines[idx] = "parameters " + " ".join(new_pairs)
    
    def _convert_projection_action(self, action_dict: Dict, netlist_text: str) -> List[Dict]:
        """Percentage action -> [{action_id, op, seq_idx, group_id, target_id, param, from_value, to_value}],
        with from_value read from netlist_text."""
        direction = action_dict['direction']  # 'increase' or 'decrease'
        target_id = action_dict['target_id']  # e.g., 'X0.M3' or '@22'
        param_token = str(action_dict['param']).strip()
        try:
            percentage = float(action_dict['percentage'])  # e.g., 15.0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid percentage '{action_dict.get('percentage')}'. Must be a finite number."
            ) from exc
        if not math.isfinite(percentage):
            raise ValueError(f"Invalid percentage '{percentage}'. Must be a finite number.")
        
        # Convert @handle to instance name if needed
        if target_id.startswith('@'):
            if target_id in self.handle_to_id:
                target_id = self.handle_to_id[target_id]
            else:
                raise ValueError(f"Handle {target_id} not found in handle_to_id mapping")
        
        # Extract current value from netlist
        lines = netlist_text.splitlines()
        inst_local = target_id.split('.')[-1]  # 'M3' from 'X0.M3'
        line_idx = self._find_inst_line_idx(lines, inst_local)
        
        if line_idx is None:
            raise ValueError(f"Instance {inst_local} not found in netlist")
        
        param_map = self._param_map(lines[line_idx])
        resolved_param = self._resolve_param_on_instance_line(param_token, param_map)
        if resolved_param is None:
            available_keys = ", ".join(param_map.keys())
            available_values = ", ".join([v for v in param_map.values() if isinstance(v, str)])
            raise ValueError(
                f"Parameter token '{param_token}' not found for instance {inst_local}. "
                f"Available keys: [{available_keys}] | referenced values: [{available_values}]"
            )
        current_value = param_map.get(resolved_param)
        
        # Check if it's a macro reference
        _, params_map, _ = self._parse_parameters_block(lines)
        macro_key_match = None
        if re.match(r'^[A-Za-z_]\w*$', current_value):
            macro_key_match = self._lookup_ci_key(params_map, current_value)
        if macro_key_match is not None:
            # It's a macro - get actual value
            from_value_str = params_map[macro_key_match]
        else:
            from_value_str = current_value
        
        # Parse current value to float
        from_value_numeric = _parse_si_simple(from_value_str)
        if not math.isfinite(from_value_numeric):
            raise ValueError(
                f"Current parameter value '{from_value_str}' for {target_id}.{resolved_param} is non-finite."
            )
        
        # Compute new value based on direction and percentage
        if direction == 'increase':
            to_value_numeric = from_value_numeric * (1 + percentage / 100.0)
        elif direction == 'decrease':
            to_value_numeric = from_value_numeric * (1 - percentage / 100.0)
        else:
            raise ValueError(f"Invalid direction: {direction}. Expected 'increase' or 'decrease'")
        if not math.isfinite(to_value_numeric):
            raise ValueError(
                f"Projected value became non-finite for action {direction} {target_id} "
                f"{resolved_param} {percentage}% from {from_value_str}."
            )
        
        # Format new value with appropriate SI unit
        to_value_str = self._format_si(to_value_numeric)
        
        # Return normalized action dict
        return [{
            "action_id": str(uuid.uuid4()),
            "op": "set_param",
            "seq_idx": 0,
            "group_id": None,
            "target_id": target_id,
            "param": resolved_param,
            "from_value": from_value_str,
            "to_value": to_value_str,
        }]

    def _convert_absolute_param_action(self, action_dict: Dict, netlist_text: str) -> List[Dict]:
        """
        Convert absolute assignment action to normalized action dict.

        Projection format: {param, value}
        Normalized format: {action_id, op, target_id, param, from_value, to_value, apply_scope}
        """
        param_token = str(action_dict['param']).strip()
        to_value_str = str(action_dict.get('value', action_dict.get('to_value'))).strip()
        if not param_token:
            raise ValueError("Missing parameter token for absolute action")
        if not to_value_str:
            raise ValueError("Missing assignment value for absolute action")

        lines = netlist_text.splitlines()
        _, params_map, _ = self._parse_parameters_block(lines)

        # Preferred path: update macro parameter directly from parameters block.
        macro_key_match = self._lookup_ci_key(params_map, param_token)
        if macro_key_match is not None:
            from_value_str = params_map[macro_key_match]
            return [{
                "action_id": str(uuid.uuid4()),
                "op": "set_param",
                "seq_idx": 0,
                "group_id": None,
                "target_id": "parameters",
                "param": macro_key_match,
                "from_value": from_value_str,
                "to_value": to_value_str,
                "apply_scope": "parameters",
            }]

        # Fallback path: resolve against instance lines when no macro key exists.
        matches = []
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.lower().startswith("parameters "):
                continue
            pm = self._param_map(line)
            if not pm:
                continue
            resolved_param = self._resolve_param_on_instance_line(param_token, pm)
            if resolved_param is None:
                continue
            inst_local = line.split()[0]
            matches.append((line_idx, inst_local, resolved_param, pm.get(resolved_param)))

        if not matches:
            raise ValueError(
                f"Parameter token '{param_token}' was not found in parameters block or instance lines."
            )
        if len(matches) > 1:
            target_preview = ", ".join(sorted({m[1] for m in matches[:5]}))
            if len(matches) > 5:
                target_preview += ", ..."
            raise ValueError(
                f"Parameter token '{param_token}' is ambiguous across multiple instances: {target_preview}. "
                "Use a unique parameter token from ACTION_SPACE."
            )

        _, inst_local, resolved_param, from_value_str = matches[0]
        return [{
            "action_id": str(uuid.uuid4()),
            "op": "set_param",
            "seq_idx": 0,
            "group_id": None,
            "target_id": inst_local,
            "param": resolved_param,
            "from_value": from_value_str,
            "to_value": to_value_str,
            "apply_scope": "instance",
        }]
    
    def _format_si(self, value: float) -> str:
        """Value as an SI token with a 0.1-resolution mantissa (e.g. 1.5e-6 -> '1.5u')."""
        if not math.isfinite(value):
            raise ValueError(f"Cannot format non-finite value as SI token: {value}")
        if value == 0:
            return "0"

        abs_val = abs(value)
        # SI suffix by magnitude
        if abs_val >= 1e9:
            unit, scale = "g", 1e9
        elif abs_val >= 1e6:
            unit, scale = "meg", 1e6
        elif abs_val >= 1e3:
            unit, scale = "k", 1e3
        elif abs_val >= 1:
            unit, scale = "", 1.0
        elif abs_val >= 1e-3:
            unit, scale = "m", 1e-3
        elif abs_val >= 1e-6:
            unit, scale = "u", 1e-6
        elif abs_val >= 1e-9:
            unit, scale = "n", 1e-9
        elif abs_val >= 1e-12:
            unit, scale = "p", 1e-12
        else:
            unit, scale = "f", 1e-15

        mantissa = value / scale
        # Quantize mantissa to 0.1 resolution for UI consistency
        mantissa_q = round(mantissa * 10.0) / 10.0
        return f"{mantissa_q:g}{unit}"
