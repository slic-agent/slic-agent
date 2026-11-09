"""
Action validation for CircuitOp environment.

Validates actions against the <ACTION_SPACE> section from the dataset/observation.
"""
import re
import math
from typing import Dict, List, Tuple, Optional


class ActionValidator:
    """
    Validates actions against allowed targets and parameters.
    
    Parses <ACTION_SPACE> section to extract:
    - valid_target_ids: Specific device IDs that can be modified
    - params_by_type: Mapping of device types to allowed parameters
    """
    
    def __init__(self):
        self.valid_targets = []  # List of valid targets (@handles or device_ids)
        self.params_by_type = {}  # {device_type: [param1, param2, ...]}
        self.params_by_target = {}  # {target_id_or_handle: [param_name, ...]}
        self.device_type_map = {}  # {device_id: device_type} from GRAPH_NODE_IDMAP
        self.handle_to_id = {}  # {@handle: device_id} mapping
        self.id_to_handle = {}  # {device_id: @handle} mapping
        self.constraints = {}  # Action constraints (percentage_min, percentage_max)
        self.action_format = ""  # Format description
        self.action_change_mode = "percentage"  # "percentage" | "absolute"
        self._targets_loaded = False
        self._use_handles = False  # Track if using handles (xml) or device_ids (netlist)
        self.local_target_map = {}  # {local_instance_name: canonical_device_id} when unique
        self.ambiguous_local_targets = set()  # local instance names that map to multiple ids
        self.prompt_target_limit = 20

    @staticmethod
    def _coerce_allowed_params(raw_params) -> List[str]:
        """Normalize dataset-provided parameter specs into a clean list of tokens."""
        if raw_params is None:
            return []
        if isinstance(raw_params, str):
            raw_items = [raw_params]
        elif hasattr(raw_params, "tolist") and not isinstance(raw_params, dict):
            converted = raw_params.tolist()
            if isinstance(converted, list):
                raw_items = converted
            else:
                raw_items = [converted]
        elif isinstance(raw_params, (list, tuple, set)):
            raw_items = list(raw_params)
        else:
            return []

        normalized = []
        for item in raw_items:
            if item is None:
                continue
            token = str(item).strip()
            if token:
                normalized.append(token)
        return normalized

    def _resolve_target_allowed_params(self, target_id: str, lookup_id: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
        """Resolve target-specific parameters or report a malformed action-space entry."""
        target_params = self._coerce_allowed_params(self.params_by_target.get(target_id))
        if target_id in self.params_by_target and not target_params:
            return [], (
                f"ACTION_SPACE integrity error: target '{target_id}' has an empty params_by_target entry. "
                "This dataset row is malformed and should be regenerated."
            )
        if target_params:
            return target_params, None

        if lookup_id and lookup_id != target_id:
            lookup_params = self._coerce_allowed_params(self.params_by_target.get(lookup_id))
            if lookup_id in self.params_by_target and not lookup_params:
                return [], (
                    f"ACTION_SPACE integrity error: target '{target_id}' maps to '{lookup_id}', "
                    "but that params_by_target entry is empty. This dataset row is malformed and should be regenerated."
                )
            if lookup_params:
                return lookup_params, None

        if self.params_by_target:
            return [], (
                f"ACTION_SPACE integrity error: target '{target_id}' has no params_by_target entry. "
                "This dataset row is malformed and should be regenerated."
            )

        return [], None
    
    def load_from_dataset(self, admissible_actions_dict: Dict, structure_xml: str, representation: str = 'xml'):
        """
        Load admissible actions from dataset field.
        
        Args:
            admissible_actions_dict: Dict with 'xml' and 'netlist' formats
            structure_xml: Structure XML containing GRAPH_NODE_IDMAP for device type mapping
            representation: 'xml' (uses @handles) or 'netlist' (uses device_ids)
        
        Example admissible_actions_dict:
        {
            "xml": {
                "valid_targets": ["@22", "@32", ...],
                "params_by_type": {"n_mosfet": ["w", "l"], ...},
                "action_format": "increase|decrease <handle> <param> <percentage>%",
                "constraints": {"percentage_min": 1, "percentage_max": 500}
            },
            "netlist": {
                "valid_targets": ["X0.M3", "X0.M5", ...],
                "params_by_type": {"n_mosfet": ["w", "l"], ...},
                "action_format": "increase|decrease <device_id> <param> <percentage>%",
                "constraints": {"percentage_min": 1, "percentage_max": 500}
            }
        }
        """
        # Parse GRAPH_NODE_IDMAP to build device_id → device_type mapping
        self._parse_nodeidmap(structure_xml)
        
        # Also parse handle ↔ device_id mappings from GRAPH_NODE_IDMAP
        self._parse_handle_mappings(structure_xml)
        
        # Select format based on representation
        if representation not in admissible_actions_dict:
            raise ValueError(f"Representation '{representation}' not found in admissible_actions")
        
        format_data = admissible_actions_dict[representation]
        
        self.valid_targets = [str(target) for target in format_data['valid_targets']]
        self.params_by_type = {
            str(device_type): self._coerce_allowed_params(params)
            for device_type, params in format_data['params_by_type'].items()
        }
        self.params_by_target = {
            str(target): self._coerce_allowed_params(params)
            for target, params in (format_data.get('params_by_target', {}) or {}).items()
        }
        self.action_format = format_data['action_format']
        self.constraints = format_data['constraints']
        self._use_handles = (representation == 'xml')

        if self.params_by_target:
            malformed_targets = []
            for target in self.valid_targets:
                lookup_id = self.handle_to_id.get(target, target)
                allowed_params, error_msg = self._resolve_target_allowed_params(target, lookup_id)
                if error_msg is not None or not allowed_params:
                    malformed_targets.append(error_msg or target)
                if len(malformed_targets) >= 5:
                    break
            if malformed_targets:
                raise ValueError(
                    "Malformed admissible_actions params_by_target entries detected: "
                    + " | ".join(malformed_targets)
                )

        self._targets_loaded = True

        self.local_target_map = {}
        self.ambiguous_local_targets = set()
        if not self._use_handles:
            local_to_targets = {}
            for target in self.valid_targets:
                local_name = str(target).split('.')[-1]
                local_to_targets.setdefault(local_name, []).append(str(target))
            for local_name, targets in local_to_targets.items():
                if len(targets) == 1:
                    self.local_target_map[local_name] = targets[0]
                else:
                    self.ambiguous_local_targets.add(local_name)

    def resolve_target_id(self, target_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a user-supplied target id to the canonical action-space target."""
        target_id = str(target_id).strip()
        if not target_id:
            return None, "Missing 'target_id' in action"
        if target_id in self.valid_targets:
            return target_id, None
        if not self._use_handles and '.' not in target_id:
            if target_id in self.ambiguous_local_targets:
                return None, (
                    f"Target alias '{target_id}' is ambiguous. Please use the hierarchical component name "
                    f"(for example X0.{target_id})."
                )
            canonical = self.local_target_map.get(target_id)
            if canonical:
                return canonical, None
        return None, None
    
    def _parse_nodeidmap(self, xml_text: str):
        """
        Parse GRAPH_NODE_IDMAP from structure XML to build device_id → device_type mapping.
        
        Format:
        <GRAPH_NODE_IDMAP>
        type=n_mosfet: @22=X0.M3 @32=X0.M5 @37=X0.M6
        type=p_mosfet: @27=X0.M4 @59=X1.M2
        type=resistor: @5=R0
        type=capacitor: @3=C0
        ...
        </GRAPH_NODE_IDMAP>
        
        Args:
            xml_text: Full XML containing GRAPH_NODE_IDMAP section
        """
        # Extract GRAPH_NODE_IDMAP section
        match = re.search(r'<GRAPH_NODE_IDMAP>(.*?)</GRAPH_NODE_IDMAP>', xml_text, re.DOTALL)
        if not match:
            return
        
        nodeidmap_content = match.group(1).strip()
        
        # Parse each line: "type=device_type: @handle=device_id @handle=device_id ..."
        for line in nodeidmap_content.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Split into type declaration and device mappings
            type_part, devices_part = line.split(':', 1)
            
            # Extract device type (e.g., "type=n_mosfet" → "n_mosfet")
            type_match = re.match(r'type=([\w_]+)', type_part.strip())
            if not type_match:
                continue
            
            device_type = type_match.group(1)
            
            # Include all types - verification happens in is_valid_action
            # Extract all device IDs (e.g., "@22=X0.M3 @32=X0.M5" → ["X0.M3", "X0.M5"])
            device_mappings = re.findall(r'@\d+=([\w\.]+)', devices_part)
            
            # Build mapping
            for device_id in device_mappings:
                self.device_type_map[device_id] = device_type
    
    def _parse_handle_mappings(self, xml_text: str):
        """
        Parse GRAPH_NODE_IDMAP to build handle ↔ device_id bidirectional mappings.
        
        Args:
            xml_text: Full XML containing GRAPH_NODE_IDMAP section
        """
        match = re.search(r'<GRAPH_NODE_IDMAP>(.*?)</GRAPH_NODE_IDMAP>', xml_text, re.DOTALL)
        if not match:
            return
        
        nodeidmap_content = match.group(1).strip()
        
        # Parse all @handle=device_id mappings
        for line in nodeidmap_content.split('\n'):
            if ':' not in line:
                continue
            _, devices_part = line.split(':', 1)
            # Extract @handle=device_id pairs
            for match in re.finditer(r'(@\d+)=([\w\.]+)', devices_part):
                handle, device_id = match.groups()
                self.handle_to_id[handle] = device_id
                self.id_to_handle[device_id] = handle

    def set_action_change_mode(self, mode: str) -> None:
        """Set action-change mode used by validation and action-space text."""
        normalized = str(mode or "percentage").strip().lower()
        if normalized not in {"percentage", "absolute"}:
            raise ValueError(
                f"Invalid action_change_mode '{mode}'. Expected 'percentage' or 'absolute'."
            )
        self.action_change_mode = normalized

    def _get_allowed_param_tokens(self) -> List[str]:
        """Get de-duplicated allowed parameter tokens in stable order."""
        ordered: List[str] = []
        seen = set()

        if self.params_by_target:
            for target in self.valid_targets:
                lookup_id = self.handle_to_id.get(target, target)
                allowed_params, _ = self._resolve_target_allowed_params(target, lookup_id)
                if not allowed_params:
                    continue
                for token in self._coerce_allowed_params(allowed_params):
                    key = token.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    ordered.append(token)
        else:
            for plist in self.params_by_type.values():
                for token in self._coerce_allowed_params(plist):
                    key = token.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    ordered.append(token)
        return ordered

    def _is_valid_absolute_param_action(self, action_dict: Dict) -> Tuple[bool, Optional[str]]:
        """Validate absolute assignment action: {'param': ..., 'value': ...}."""
        param = str(action_dict.get('param', '')).strip()
        value = action_dict.get('value', action_dict.get('to_value'))
        value = "" if value is None else str(value).strip()

        if not param:
            return False, "Missing 'param' in action"
        if not value:
            return False, "Missing 'value' in action"

        allowed_params = self._get_allowed_param_tokens()
        normalized_param = self._normalize_param_token(param, allowed_params)
        if normalized_param is None:
            preview = ", ".join(allowed_params[:20]) if allowed_params else "(none)"
            if len(allowed_params) > 20:
                preview += ", ..."
            return False, (
                f"Parameter '{param}' is not an admissible parameter token. "
                f"Allowed parameter tokens: {preview}"
            )

        if not self._looks_like_numeric_token(value):
            return False, (
                f"Invalid value '{value}' for parameter '{normalized_param}'. "
                "Use a numeric token with optional SI suffix (for example 420n, 1.2u, 3k)."
            )

        action_dict['param'] = normalized_param
        action_dict['value'] = value
        return True, None
    
    def is_valid_action(self, action_dict: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check if an action is valid.
        
        Validation depends on action_change_mode:
        - percentage: target/param/direction/percentage checks
        - absolute:   param/value assignment checks
        
        Args:
            action_dict: Dict with 'direction', 'target_id', 'param', 'percentage'
        
        Returns:
            (is_valid, error_message) tuple
        """
        if not self._targets_loaded:
            return False, "Action validator not initialized - no targets loaded from dataset"

        if self.action_change_mode == 'absolute':
            if ('direction' in action_dict or 'percentage' in action_dict) and 'value' not in action_dict and 'to_value' not in action_dict:
                tokens = self._get_allowed_param_tokens()
                example = f"{tokens[0]}=420n" if tokens else "w=420n"
                return False, ("Percentage actions (<direction> <target> <param> <percentage>%) are accepted only with action_change_mode: "
                               f"percentage; use <param_name>=<value>, e.g. {example}")
            return self._is_valid_absolute_param_action(action_dict)
        if 'value' in action_dict and not action_dict.get('direction'):
            return False, ("Assignment actions (<param>=<value>) are accepted only with action_change_mode: absolute; "
                           "use <direction> <target> <param> <percentage>%, e.g. increase @22 w 15%")
        
        # Check required fields
        target_id = action_dict.get('target_id', '')
        param = action_dict.get('param', '')
        direction = action_dict.get('direction', '')
        percentage = action_dict.get('percentage')
        
        if not target_id:
            return False, "Missing 'target_id' in action"
        if not param:
            return False, "Missing 'param' in action"
        if not direction:
            return False, "Missing 'direction' in action"
        if percentage is None:
            return False, "Missing 'percentage' in action"
        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            return False, f"Invalid percentage '{percentage}'. Must be a finite number."
        if not math.isfinite(percentage):
            return False, f"Invalid percentage '{percentage}'. Must be a finite number."
        action_dict['percentage'] = percentage
        
        # Validate direction
        if direction not in ['increase', 'decrease']:
            return False, f"Invalid direction '{direction}'. Must be 'increase' or 'decrease'"
        
        # Validate percentage constraints
        pct_min = self.constraints.get('percentage_min', 1)
        pct_max = self.constraints.get('percentage_max', 500)
        if pct_min is not None and percentage < pct_min:
            return False, f"Percentage {percentage}% below minimum {pct_min}%"
        if pct_max is not None and percentage > pct_max:
            return False, f"Percentage {percentage}% above maximum {pct_max}%"
        
        canonical_target, canonical_error = self.resolve_target_id(target_id)
        if canonical_error is not None:
            return False, canonical_error
        if canonical_target is not None:
            target_id = canonical_target

        # Check if target_id is in valid_targets
        # Also check alternate format (handle <-> device_id) and give helpful message
        alternate_target = None
        if target_id not in self.valid_targets:
            # Check if user provided alternate format
            if self._use_handles and not target_id.startswith('@'):
                # Expected handle, got device_id - check if there's a matching handle
                alternate_target = self.id_to_handle.get(target_id)
                if alternate_target and alternate_target in self.valid_targets:
                    return False, (
                        f"Target '{target_id}' is a device ID, but this environment expects handles. "
                        f"Please use '{alternate_target}' instead. "
                        f"Valid handle targets: {', '.join(self.valid_targets[:10])}"
                        + ("..." if len(self.valid_targets) > 10 else "")
                    )
            elif not self._use_handles and target_id.startswith('@'):
                # Expected device_id, got handle - convert and check
                alternate_target = self.handle_to_id.get(target_id)
                if alternate_target and alternate_target in self.valid_targets:
                    return False, (
                        f"Target '{target_id}' is a handle, but this environment expects device IDs. "
                        f"Please use '{alternate_target}' instead. "
                        f"Valid device ID targets: {', '.join(self.valid_targets[:10])}"
                        + ("..." if len(self.valid_targets) > 10 else "")
                    )
            
            # Neither format is valid
            return False, (
                f"Target '{target_id}' is not a valid changeable device. "
                f"Valid targets: {', '.join(self.valid_targets[:10])}"
                + ("..." if len(self.valid_targets) > 10 else "")
            )
        
        # Convert @handle to device_id if needed for type lookup
        if target_id.startswith('@'):
            lookup_id = self.handle_to_id.get(target_id)
            if lookup_id is None:
                return False, f"Handle '{target_id}' not found in handle mapping"
        else:
            # Direct device_id (e.g., X0.M3)
            lookup_id = target_id
        
        # Infer device type from lookup_id
        device_type = self.device_type_map.get(lookup_id)
        if device_type is None:
            return False, f"Could not determine device type for '{lookup_id}'"
        
        # Check if device type has parameter definitions
        if device_type not in self.params_by_type:
            return False, (
                f"Device type '{device_type}' not recognized. "
                f"Valid types: {', '.join(self.params_by_type.keys())}"
            )
        
        # Check if parameter is allowed for this target. Rows without
        # params_by_target fall back to device-type params.
        allowed_params, target_param_error = self._resolve_target_allowed_params(target_id, lookup_id)
        if target_param_error is not None:
            return False, target_param_error
        if not allowed_params:
            allowed_params = self.params_by_type.get(device_type, [])
        normalized_param = self._normalize_param_token(param, allowed_params)
        if normalized_param is None:
            # Check if user accidentally used a device type instead of parameter
            param_lc = str(param).lower()
            if param_lc in {t.lower() for t in self.params_by_type.keys()}:
                return False, (
                    f"ERROR: '{param}' is a device TYPE, not a parameter. "
                    f"For {device_type} devices, use one of these parameters: {', '.join(allowed_params)}. "
                    f"Example: <action>increase {target_id} {allowed_params[0]} 10%</action>"
                )
            matching_targets = []
            for candidate_target, candidate_params in self.params_by_target.items():
                if self._normalize_param_token(param, candidate_params) is not None:
                    matching_targets.append(candidate_target)
            if matching_targets:
                preview = ", ".join(matching_targets[:5])
                if len(matching_targets) > 5:
                    preview += ", ..."
                return False, (
                    f"Parameter '{param}' not allowed for {device_type} target '{target_id}'. "
                    f"Allowed for this target: {', '.join(allowed_params)}. "
                    f"This parameter is available on: {preview}"
                )
            return False, (
                f"Parameter '{param}' not allowed for {device_type}. "
                f"Allowed: {', '.join(allowed_params)}"
            )
        
        return True, None

    @staticmethod
    def _looks_like_numeric_token(token: str) -> bool:
        token = str(token).strip()
        if not token:
            return False
        return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[A-Za-z%]*", token))

    @classmethod
    def _extract_param_candidates(cls, param: str) -> List[str]:
        raw = str(param).strip()
        if not raw:
            return []

        parts = [p.strip() for p in raw.split('=') if p.strip()]
        non_numeric_parts = [p for p in parts if not cls._looks_like_numeric_token(p)]

        ordered: List[str] = []
        if len(non_numeric_parts) >= 2:
            first = non_numeric_parts[0]
            second = non_numeric_parts[1]
            if len(first) <= 2 and len(second) > len(first):
                ordered.extend([second, first])
                ordered.extend(non_numeric_parts[2:])
            else:
                ordered.extend(non_numeric_parts)
        else:
            ordered.extend(non_numeric_parts)

        ordered.append(raw)

        seen = set()
        deduped = []
        for candidate in ordered:
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    @classmethod
    def _normalize_param_token(cls, param: str, allowed_params: List[str]) -> Optional[str]:
        allowed_params = cls._coerce_allowed_params(allowed_params)
        if not allowed_params:
            return None
        allowed_params_lc = {str(p).lower(): str(p) for p in allowed_params}
        for candidate in cls._extract_param_candidates(param):
            matched = allowed_params_lc.get(candidate.lower())
            if matched is not None:
                return matched
            candidate_lc = candidate.lower()
            if candidate_lc in {'w', 'l', 'r', 'c', 'dc'}:
                suffix_matches = [
                    str(p) for p in allowed_params
                    if (
                        str(p).lower().endswith(f"_{candidate_lc}")
                        or str(p).lower().startswith(f"{candidate_lc}_")
                    )
                ]
                if len(suffix_matches) == 1:
                    return suffix_matches[0]
        return None

    def set_percentage_constraints(self, enforce: bool, pct_min: Optional[float] = None, pct_max: Optional[float] = None):
        """Update percentage range constraints for validation/prompt text."""
        if not enforce:
            self.constraints['percentage_min'] = None
            self.constraints['percentage_max'] = None
            return

        if pct_min is not None:
            self.constraints['percentage_min'] = float(pct_min)
        if pct_max is not None:
            self.constraints['percentage_max'] = float(pct_max)

    def set_prompt_target_limit(self, max_targets: Optional[int]) -> None:
        """Set max number of target rows shown in admissible-actions prompt text."""
        if max_targets is None:
            self.prompt_target_limit = 20
            return
        try:
            parsed = int(max_targets)
        except (TypeError, ValueError):
            self.prompt_target_limit = 20
            return
        self.prompt_target_limit = parsed
    
    def get_admissible_actions_text(self) -> List[str]:
        """Admissible-action lines shown in the prompt."""
        if not self._targets_loaded:
            return ["Action validation not available - targets not loaded"]
        
        actions = []
        example_fallback_target = '@X' if self._use_handles else 'X0.MM0'

        if self.action_change_mode == 'absolute':
            allowed_tokens = self._get_allowed_param_tokens()
            example_param = allowed_tokens[0] if allowed_tokens else "w"
            example_param2 = allowed_tokens[1] if len(allowed_tokens) > 1 else example_param
            actions.append("Action format: <action><param_name>=<value>; <param_name>=<value></action>")
            actions.append(
                f"  Example: <action>{example_param}=420n; {example_param2}=450n</action>"
            )
            actions.append(
                "  Use one or more assignments separated by semicolons in a single <action> block."
            )
            actions.append(
                "  Values must be numeric with optional SI suffix (f, p, n, u, m, k, meg, g)."
            )
        else:
            # Action format with clear explanation
            if self._use_handles:
                actions.append(f"Action format: <action>{self.action_format}</action>")
            else:
                actions.append(
                    "Action format: <action>increase|decrease <component_name> <param_name> <percentage>%</action>"
                )
                actions.append(
                    "  Component name can be hierarchical (for example X0.MM6) or a unique local ID (for example MM6)."
                )
            example_target = self.valid_targets[0] if len(self.valid_targets) > 0 else example_fallback_target
            example_param = None
            if self.params_by_target:
                example_param = self._coerce_allowed_params(self.params_by_target.get(example_target))
                if not example_param and example_target.startswith('@'):
                    device_id = self.handle_to_id.get(example_target)
                    example_param = self._coerce_allowed_params(self.params_by_target.get(device_id, []))
                if len(example_param) > 0:
                    example_param = example_param[0]
                else:
                    example_param = None
            if not example_param and self.params_by_type:
                first_params = next(iter(self.params_by_type.values()), [])
                example_param = first_params[0] if len(first_params) > 0 else 'w'
            example_param = example_param or 'w'
            actions.append(
                f"  Example: <action>increase {example_target} {example_param} 15%</action> changes "
                f"{example_target} via parameter '{example_param}' by 15%"
            )

            if self.constraints:
                pct_min = self.constraints.get('percentage_min', 1)
                pct_max = self.constraints.get('percentage_max', 500)
                if pct_min is not None or pct_max is not None:
                    if pct_min is not None and pct_max is not None:
                        actions.append("Percentage range: " + f"{pct_min}-{pct_max}%")
                        actions.append(f"  Example: Valid percentages are from {pct_min}% to {pct_max}%, inclusive, using 1% increments.")
                        # one action per parameter per turn; repeated actions are a format error.
                        actions.append(f"  Rule: each parameter may appear at most once per turn and the same action must not be repeated. "
                                       f"A larger change is made over successive turns ({pct_max}% per turn at most), not by listing the action several times.")
                    elif pct_min is not None:
                        actions.append("Percentage minimum: " + f"{pct_min}%")
                    elif pct_max is not None:
                        actions.append("Percentage maximum: " + f"{pct_max}%")
        
        # Group by device type and show allowed parameters
        if self.action_change_mode == 'absolute':
            allowed_tokens = self._get_allowed_param_tokens()
            preview = ", ".join(allowed_tokens[:60]) if allowed_tokens else "(none)"
            if len(allowed_tokens) > 60:
                preview += ", ..."
            actions.append("\nAllowed parameter tokens:")
            actions.append(f"  {preview}")
        elif self.params_by_target:
            if self._use_handles:
                actions.append("Allowed valid target handles and parameter tokens:")
            else:
                actions.append("Allowed valid target components and parameter tokens:")
            if self.prompt_target_limit is None or self.prompt_target_limit <= 0:
                preview_targets = self.valid_targets
            else:
                preview_targets = self.valid_targets[:self.prompt_target_limit]
            for target in preview_targets:
                plist = self._coerce_allowed_params(self.params_by_target.get(target))
                if not plist and target.startswith('@'):
                    device_id = self.handle_to_id.get(target)
                    plist = self._coerce_allowed_params(self.params_by_target.get(device_id, []))
                if len(plist) > 0:
                    actions.append(f"  {target}: {', '.join(plist)}")
            if len(self.valid_targets) > len(preview_targets):
                actions.append(
                    "  [truncated target list: showing "
                    f"{len(preview_targets)} of {len(self.valid_targets)} targets; "
                    "set action_space_prompt_max_targets<=0 in reward config to show all]"
                )
        else:
            actions.append("\nAllowed parameters by device type:")
            for device_type, params in self.params_by_type.items():
                params_str = ", ".join(params)
                actions.append(f"  {device_type}: {params_str}")
            actions.append("  Example: For n_mosfet, you can modify 'w' (width) or 'l' (length)")
        
        if self._use_handles:
            actions.append(
                "\nIMPORTANT: Use a parameter token from the action space. "
                "If FEATURES shows enriched values like w=w_MM30=419.5n, the actionable token is w_MM30 "
                "(or w when that is what the target allows), not the device type."
            )
        
        return actions
