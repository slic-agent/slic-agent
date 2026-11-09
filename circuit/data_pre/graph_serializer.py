import math
import re

import networkx as nx


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _format_gain_db(value):
    if not _is_finite_number(value):
        return None
    return f"{float(value):.1f} dB"


def _format_db_tenth(value):
    if not _is_finite_number(value):
        return None
    return f"{float(value):.1f} dB"


def _format_power_uw_from_watts(value_w):
    if not _is_finite_number(value_w):
        return None
    return f"{int(round(float(value_w) * 1e6))} uW"


def _format_phase_margin_deg(value):
    if not _is_finite_number(value):
        return None
    return f"{int(round(float(value)))} deg"


def _format_scaled_value(value, units):
    abs_value = abs(float(value))
    chosen_scale, chosen_unit = units[-1]
    for scale, unit in units:
        if abs_value >= scale:
            chosen_scale, chosen_unit = scale, unit
            break
    return f"{int(round(float(value) / chosen_scale))} {chosen_unit}"


def _format_bandwidth_human(value_hz):
    if not _is_finite_number(value_hz) or float(value_hz) <= 0:
        return None
    return _format_scaled_value(
        float(value_hz),
        [
            (1e9, "GHz"),
            (1e6, "MHz"),
            (1e3, "kHz"),
            (1.0, "Hz"),
        ],
    )


def _format_offset_human(value_v):
    if not _is_finite_number(value_v):
        return None
    return _format_scaled_value(
        float(value_v),
        [
            (1.0, "V"),
            (1e-3, "mV"),
            (1e-6, "uV"),
            (1e-9, "nV"),
        ],
    )


class GraphXMLGenerator:
    # --- Netlist parser constants ---
    MOSFET_TERMINALS_STD = ['D', 'G', 'S', 'B']
    SOURCE_TERMINALS_STD = ['P', 'N']
    NODE_PATTERN_RE = re.compile(r'\((.*?)\)')
    SC_PATTERN = re.compile(r"(\w+)=([\w\.\+\-\*/]+)")

    def __init__(self, pin_names=['vout', 'vb_in', 'vin', 'vbg', 'vdd', 'vss', '0']):
        self.pin_names_lc = [name.lower() for name in pin_names]
        self.parsed_subcircuits = {}

    def generate_static_structure(self, netlist_str, topo_name):
        """
        Extracts the fixed topology. Returns (xml_string, handle2id, nodes_data):
        - xml_string: the formatted XML structure
        - handle2id: dict mapping @N → node_id (e.g., {"@0": "0", "@1": "vdd"})
        - nodes_data: list of node dicts with 'id' field set to full node name
        """
        G = self.parse_spectre_to_graph(netlist_str)
        nodes_data = []
        for n, d in G.nodes(data=True):
            # ✅ Store full node name as ID (matches NetworkX json_graph format)
            d['id'] = n
            nodes_data.append(d)
            
        raw_ids = [n['id'] for n in nodes_data]
        id2handle, handle2id = self._make_idmaps(raw_ids)
        
        blocks = [
            f"<GRAPH_INFO>\nTOPOLOGY_NAME: {topo_name}\n</GRAPH_INFO>\n",
            self._emit_idmap(handle2id, nodes_data),
            self._emit_pinmap(id2handle, raw_ids),
            self._emit_edges(G, id2handle)
        ]
        return "".join(blocks), handle2id, nodes_data

    def scan_instance_params_aligned(self, netlist_str, G):
        """
        Scan netlist for instance parameters and align them to graph node names.
        Resolves parameter references (e.g., M1_w) to actual values from parameters line.
        
        Args:
        - netlist_str: Raw netlist string
        - G: NetworkX graph from parse_spectre_to_graph
        
        Returns:
        - dict {graph_node_name: {param: value, ...}}
          Example: {"X0.M1": {"w": "1.5u", "l": "180n"}, "R0": {"r": "10k"}}
        """
        # 0) Extract parameter definitions from netlist
        param_defs = self._extract_parameter_definitions(netlist_str)
        
        # 1) Scan raw netlist for local instance names
        raw_params = {}
        for ln in netlist_str.splitlines():
            ln = ln.strip()
            # ✅ Skip comments, parameters, and simulator directives
            if not ln or ln.startswith('//') or ln.startswith('#') or ln.lower().startswith(('subckt', 'ends', 'parameters', 'simulator', 'global', 'include')):
                continue
            parts = ln.split()
            if not parts:
                continue
            inst = parts[0]
            
            # Extract key=value pairs from the line
            tail = ln.split(')', 1)[-1] if ')' in ln else ln
            kvs = {k.lower(): v for k, v in self.SC_PATTERN.findall(tail)}
            
            # Resolve parameter references while preserving the actionable parameter token
            # in the emitted FEATURES block. Example: `w=Mpair_w` -> `w=Mpair_w=1.5u`.
            if kvs:
                resolved_kvs = {}
                for k, raw_v in kvs.items():
                    evaluated_v = self._evaluate_expression(raw_v, param_defs) if param_defs else raw_v
                    if raw_v != evaluated_v and re.fullmatch(r'[A-Za-z_]\w*(?:[\./\*\+\-]\w+)*', raw_v):
                        resolved_kvs[k] = f"{raw_v}={evaluated_v}"
                    else:
                        resolved_kvs[k] = evaluated_v
                kvs = resolved_kvs
            
            if kvs:
                raw_params[inst] = kvs
        
        # 2) Align to mangled graph node names
        inst_params = {}
        for n, d in G.nodes(data=True):
            node_type = d.get('type', '')
            # Only process instance nodes
            if 'instance' in node_type or node_type in (
                'passive_component_instance', 
                'active_component_instance',
                'subcircuit_instance_node',
                'unknown_instance'
            ):
                # Extract base name: "X0.M1" → "M1", "R0" → "R0"
                base = n.split('.')[-1]
                if base in raw_params:
                    inst_params[n] = raw_params[base]
        
        return inst_params
    
    def _extract_parameter_definitions(self, netlist_str):
        """Parse the first `parameters` line into {name: value}, e.g. {"M1_w": "1.5u", "vdd": "1.2"}."""
        param_dict = {}
        
        for line in netlist_str.splitlines():
            line = line.strip()
            # Look for parameters line
            if line.lower().startswith('parameters'):
                # Extract all key=value pairs from the parameters line
                param_line = line[len('parameters'):].strip()
                kvs = self.SC_PATTERN.findall(param_line)
                for k, v in kvs:
                    param_dict[k] = v
                break
        
        return param_dict
    
    def _evaluate_expression(self, expr, param_dict):
        """Substitute parameters into a simple expression (e.g. vdd/2) and evaluate it; returns a string."""
        # Substitute parameter names (whole-word only to avoid substring collisions,
        # e.g. param w_MM4 must not match inside w_MM41).
        resolved = expr
        for param_name in sorted(param_dict.keys(), key=len, reverse=True):
            param_value = param_dict[param_name]
            resolved = re.sub(r'(?<!\w)' + re.escape(param_name) + r'(?!\w)',
                               param_value, resolved)
        
        # If no operators, return as-is
        if resolved == expr or not any(op in resolved for op in ['+', '-', '*', '/']):
            return resolved
        
        # Extract unit suffix
        unit_match = re.search(r'([munpfkMGT]?)$', resolved)
        unit = unit_match.group(1) if unit_match else ''
        numeric_part = resolved.rstrip('munpfkMGT')
        
        # Try to evaluate
        try:
            if re.match(r'^[\d\.\+\-\*/\(\)\s]+$', numeric_part):
                result = eval(numeric_part)
                if isinstance(result, float):
                    return f"{result:.4g}{unit}"
                return f"{result}{unit}"
        except:
            pass
        
        return resolved

    def generate_dynamic_state(self, metrics, nodes, inst_params, op_data, id2handle, constraint_specs=None):
        """
        Creates the Observation block (PERFORMANCE + FEATURES).

        Parameters:
        - metrics: dict with 'gain', 'power' keys
        - nodes: list of node dicts with 'id' field (from generate_static_structure)
        - inst_params: dict {device_id: {param: value}} - already aligned to graph IDs
        - op_data: dict {device_id: operating_point_data} from simulator
        - id2handle: dict mapping device_id → @handle (e.g., {"X0.M1": "@5"})
        - constraint_specs: optional set of spec names to show (e.g. {'gain', 'power'}).
                            When None, all specs are shown.
        """
        cs = set(constraint_specs) if constraint_specs is not None else None
        perf_lines = ["<PERFORMANCE>"]
        power_str = _format_power_uw_from_watts(metrics.get('power', 0) or 0)
        bw_val = metrics.get('bandwidth_MHz', 0)
        if not bw_val:
            bw_val = (metrics.get('bandwidth', 0) or metrics.get('bandwidth_Hz', 0)) / 1e6
        bw_hz = (metrics.get('bandwidth', 0) or metrics.get('bandwidth_Hz', 0) or 0.0)
        if not bw_hz and _is_finite_number(bw_val):
            bw_hz = float(bw_val) * 1e6
        gain_str = _format_gain_db(metrics.get('gain', 0))
        if cs is None or 'gain' in cs:
            perf_lines.append(f"<GainAt1kHz>{gain_str or '0.0 dB'}</GainAt1kHz>")
        if cs is None or 'power' in cs:
            perf_lines.append(f"<Power>{power_str or '0 uW'}</Power>")
        try:
            bw_num = float(bw_val)
        except (TypeError, ValueError):
            bw_num = 0.0
        if bw_num > 0 and abs(bw_num - 9999.99) > 1e-2 and (cs is None or 'bandwidth' in cs):
            bw_str = _format_bandwidth_human(bw_hz)
            if bw_str:
                perf_lines.append(f"<Bandwidth3dB>{bw_str}</Bandwidth3dB>")
        elif (metrics.get('bandwidth_MHz') is None and 'bandwidth_kind' in metrics
              and (cs is None or 'bandwidth' in cs)):
            # None = the -3 dB bandwidth is not measurable; say so instead of
            # dropping the line (metrics without 'bandwidth_kind' omit it).
            perf_lines.append("<Bandwidth3dB>not measurable</Bandwidth3dB>")
        diff_gain = metrics.get('differential_gain_db')
        if diff_gain is not None and cs is None:
            diff_gain_str = _format_gain_db(diff_gain)
            if diff_gain_str:
                perf_lines.append(f"<DifferentialGain>{diff_gain_str}</DifferentialGain>")
        offset_v = metrics.get('offset_v')
        if offset_v is not None and (cs is None or 'offset' in cs):
            offset_str = _format_offset_human(offset_v)
            if offset_str:
                perf_lines.append(f"<Offset>{offset_str}</Offset>")
        cmrr_db = metrics.get('cmrr_db')
        if cmrr_db is not None and (cs is None or 'cmrr' in cs):
            cmrr_str = _format_db_tenth(cmrr_db)
            if cmrr_str:
                perf_lines.append(f"<CMRR>{cmrr_str}</CMRR>")
        psrr_plus_db = metrics.get('psrr_plus_db')
        if psrr_plus_db is not None and (cs is None or 'psrr_plus' in cs):
            psrr_plus_str = _format_db_tenth(psrr_plus_db)
            if psrr_plus_str:
                perf_lines.append(f"<PSRRPlus>{psrr_plus_str}</PSRRPlus>")
        psrr_minus_db = metrics.get('psrr_minus_db')
        if psrr_minus_db is not None and (cs is None or 'psrr_minus' in cs):
            psrr_minus_str = _format_db_tenth(psrr_minus_db)
            if psrr_minus_str:
                perf_lines.append(f"<PSRRMinus>{psrr_minus_str}</PSRRMinus>")
        pm_val = metrics.get('phase_margin_deg')
        if pm_val is not None and (cs is None or 'phase_margin' in cs):
            pm_str = _format_phase_margin_deg(pm_val)
            if pm_str:
                perf_lines.append(f"<PhaseMargin>{pm_str}</PhaseMargin>")
        elif (metrics.get('phase_margin_status') == 'stable_no_crossing'
              and (cs is None or 'phase_margin' in cs)):
            # no 0 dB crossing + clean band -> spec passes; keep it visible
            perf_lines.append("<PhaseMargin>stable (no 0 dB crossing; spec passed)</PhaseMargin>")
        inband_db = metrics.get('inband_flatness_db')
        if inband_db is not None:
            perf_lines.append(f"<InBandFlatness>{float(inband_db):.1f} dB (limit 6)</InBandFlatness>")
        # causal Validity line (same as the env error-path renderer)
        _g_eff = metrics.get('differential_gain_db')
        if _g_eff is None:
            _g_eff = metrics.get('gain')
        try:
            _alive = _g_eff is not None and float(_g_eff) == float(_g_eff) and float(_g_eff) > 0.0
        except (TypeError, ValueError):
            _alive = False
        if not _alive:
            perf_lines.append("<Validity>non-positive gain; power/bandwidth/PM not meaningful until gain > 0</Validity>")
        elif inband_db is not None and float(inband_db) > 6.0:
            perf_lines.append(f"<Validity>in-band peaking {float(inband_db):.1f} dB exceeds 6 dB; gain/bandwidth/PM readings unreliable</Validity>")
        perf_lines.append("</PERFORMANCE>\n")

        features_xml = self._emit_features(nodes, inst_params, op_data, id2handle)
        
        return "".join(perf_lines) + features_xml

    # =========================================================================
    # CORE NETLIST PARSER LOGIC
    # =========================================================================

    def parse_spectre_to_graph(self, netlist_string):
        """Parse a Spectre netlist into a NetworkX graph, expanding sub-circuit instances."""
        self.parsed_subcircuits = {} 
        G = nx.Graph()
        lines = netlist_string.strip().split('\n')

        current_sub_name = None
        current_sub_ports = []
        current_sub_body_lines = []
        top_level_lines = []

        # Pass 1: Collect subcircuit definitions
        for line_content in lines:
            s_line = line_content.strip()
            # ✅ Filter both // and # comments, and skip 'parameters' lines
            if not s_line or s_line.startswith('//') or s_line.startswith('#') or s_line.lower().startswith('parameters'):
                continue

            if s_line.lower().startswith('subckt'):
                parts_sub = s_line.split()
                current_sub_name = parts_sub[1]
                ports = []
                for p in parts_sub[2:]:
                    if '=' in p or p.lower() == 'ends': break
                    ports.append(p)
                current_sub_ports = ports
                current_sub_body_lines = []
            elif s_line.lower().startswith('ends'):
                if current_sub_name:
                    self.parsed_subcircuits[current_sub_name] = {
                        'ports': current_sub_ports,
                        'body_lines': current_sub_body_lines
                    }
                    current_sub_name = None
            elif current_sub_name:
                current_sub_body_lines.append(s_line)
            else:
                top_level_lines.append(s_line)

        # Pass 2: Process top-level and expand
        for line in top_level_lines:
            top_level_parts = line.split()
            if not top_level_parts: continue
            
            top_instance_name = top_level_parts[0]
            top_component_char = top_instance_name[0].upper()
            top_sub_type_name = None
            top_connected_nets = []
            match_top_nodes = self.NODE_PATTERN_RE.search(line)

            if match_top_nodes:
                top_node_str = match_top_nodes.group(1)
                top_connected_nets = top_node_str.split()
                original_line_after_nodes_str = line[match_top_nodes.end():].strip()
                tokens_after_nodes = original_line_after_nodes_str.split()
                if tokens_after_nodes and tokens_after_nodes[0] in self.parsed_subcircuits and '=' not in tokens_after_nodes[0]:
                    top_sub_type_name = tokens_after_nodes[0]
            else:
                if len(top_level_parts) >= 3 and '=' not in top_level_parts[1] and '=' not in top_level_parts[2]:
                     top_connected_nets = top_level_parts[1:3]
                else: continue
            
            for net_name in top_connected_nets:
                self._ensure_node(G, net_name)

            if top_component_char == 'X' and top_sub_type_name:
                sub_def_ports = self.parsed_subcircuits[top_sub_type_name]['ports']
                port_map = {formal: top_connected_nets[i] for i, formal in enumerate(sub_def_ports) if i < len(top_connected_nets)}
                self._expand_subcircuit(G, top_instance_name + ".", top_sub_type_name, port_map)
            else:
                if top_sub_type_name and top_component_char != 'X':
                    G.add_node(top_instance_name, type='subcircuit_instance_node', sub_type=top_sub_type_name)
                    for net in top_connected_nets: G.add_edge(top_instance_name, net)
                else:
                     self._add_primitive(G, top_instance_name, top_component_char, top_connected_nets, top_level_parts)
                     
        # Final Safeguard for pin types
        for node_name in list(G.nodes()):
            if node_name.lower() in self.pin_names_lc:
                t = G.nodes[node_name].get('type','')
                if 'instance' not in t and 'terminal' not in t:
                    G.nodes[node_name]['type'] = 'pin'
        return G

    def _ensure_node(self, G, node_name):
        if not G.has_node(node_name):
            node_type = 'pin' if node_name.lower() in self.pin_names_lc else 'net'
            G.add_node(node_name, type=node_type)
        elif G.nodes[node_name].get('type') == 'net' and node_name.lower() in self.pin_names_lc:
            G.nodes[node_name]['type'] = 'pin'

    def _add_primitive(self, graph, instance_name, component_char, connected_nets, line_parts):
        for net in connected_nets: self._ensure_node(graph, net)
        graph.add_node(instance_name)

        if component_char == 'M':
            graph.nodes[instance_name]['type'] = 'active_component_instance'
            mosfet_type = 'n_mosfet' if any('nch' in p.lower() for p in line_parts) else ('p_mosfet' if any('pch' in p.lower() for p in line_parts) else 'mosfet')
            graph.nodes[instance_name]['device_type'] = mosfet_type
            for i, net in enumerate(connected_nets):
                if i < len(self.MOSFET_TERMINALS_STD):
                    term = f"{instance_name}_{self.MOSFET_TERMINALS_STD[i]}"
                    graph.add_node(term, type='active_component_terminal')
                    graph.add_edge(instance_name, term)
                    graph.add_edge(term, net)

        elif component_char in ('V', 'I'):
            source_type = 'voltage_source' if component_char == 'V' else 'current_source'
            graph.nodes[instance_name]['type'] = 'active_component_instance'
            graph.nodes[instance_name]['device_type'] = source_type
            for i, net in enumerate(connected_nets):
                if i < len(self.SOURCE_TERMINALS_STD):
                    term = f"{instance_name}_{self.SOURCE_TERMINALS_STD[i]}"
                    graph.add_node(term, type='active_component_terminal')
                    graph.add_edge(instance_name, term)
                    graph.add_edge(term, net)

        elif component_char in ('R', 'C', 'L'):
            dev_type = {'R': 'resistor', 'C': 'capacitor', 'L': 'inductor'}.get(component_char)
            graph.nodes[instance_name]['type'] = 'passive_component_instance'
            graph.nodes[instance_name]['device_type'] = dev_type
            for net in set(connected_nets): graph.add_edge(instance_name, net)

    def _expand_subcircuit(self, graph, prefix, sub_name, port_map):
        if sub_name not in self.parsed_subcircuits: return
        sub_def = self.parsed_subcircuits[sub_name]
        formal_ports = sub_def['ports']

        for internal_line in sub_def['body_lines']:
            parts = internal_line.split()
            if not parts: continue
            mangled_name = prefix + parts[0]
            comp_char = parts[0][0].upper()
            
            sub_type = None
            match = self.NODE_PATTERN_RE.search(internal_line)
            if match:
                internal_nets = match.group(1).split()
                after = internal_line[match.end():].split()
                if after and after[0] in self.parsed_subcircuits and '=' not in after[0]:
                    sub_type = after[0]
            else:
                internal_nets = parts[1:]
                if internal_nets and internal_nets[-1] in self.parsed_subcircuits:
                    sub_type = internal_nets.pop()

            mapped_nets = [port_map[n] if n in formal_ports else prefix + n for n in internal_nets]
            for m_net in mapped_nets: self._ensure_node(graph, m_net)

            if sub_type and comp_char == 'X':
                nested_ports = self.parsed_subcircuits[sub_type]['ports']
                nested_map = {formal: mapped_nets[i] for i, formal in enumerate(nested_ports) if i < len(mapped_nets)}
                self._expand_subcircuit(graph, mangled_name + ".", sub_type, nested_map)
            else:
                self._add_primitive(graph, mangled_name, comp_char, mapped_nets, parts)

    # =========================================================================
    # XML GENERATION HELPERS
    # =========================================================================

    def _emit_idmap(self, handle2id, nodes):
        """Generate GRAPH_NODE_IDMAP block, grouped by type."""
        node_map = {n['id']: n for n in nodes}
        
        # Group nodes by type
        type_groups = {}
        for h, rid in handle2id.items():
            node_data = node_map.get(rid, {})
            typ = (node_data.get("device_type") or node_data.get("type") or "unknown").lower()
            if not typ:
                typ = "unknown"
            type_groups.setdefault(typ, []).append((h, rid))
        
        # Sort types: pins first, then device types, then nets
        def type_priority(typ):
            if typ == "pin": return 0
            elif typ == "net": return 3
            elif "mosfet" in typ or "source" in typ or "resistor" in typ or "capacitor" in typ: return 1
            elif "terminal" in typ or "instance" in typ: return 2
            else: return 4
        
        sorted_types = sorted(type_groups.keys(), key=lambda t: (type_priority(t), t))
        
        lines = ["<GRAPH_NODE_IDMAP>"]
        for typ in sorted_types:
            handles_and_ids = sorted(type_groups[typ], key=lambda x: x[0])  # Sort by handle
            
            # Format as: type=n_mosfet: @5=X0.M1 @6=X0.M2 @7=X1.M0
            id_pairs = " ".join([f"{h}={rid}" for h, rid in handles_and_ids])
            lines.append(f"type={typ}: {id_pairs}")
        
        lines.append("</GRAPH_NODE_IDMAP>\n")
        return "\n".join(lines)

    def _emit_features(self, nodes, inst_params, op_data, id2handle):
        """Generate FEATURES block."""
        lines = ["<FEATURES>"]
        
        # Filter out None entries (from Parquet padding for different topology sizes)
        nodes = [n for n in nodes if n is not None]
        
        node_by_id = {n['id']: n for n in nodes}
        typ_by_id = {n['id']: (n.get("device_type") or n.get("type") or "").lower() for n in nodes}
        
        # ✅ Build features dict for ALL instance nodes, not just those with inst_params
        feats_by_id = {}
        
        # First pass: Initialize ALL instance/device nodes with type information
        for n in nodes:
            did = n['id']
            typ = typ_by_id.get(did, "")
            
            # Only include device/instance nodes (skip nets, pins, terminals)
            if not any(keyword in typ for keyword in ["instance", "mosfet", "resistor", "capacitor", "source"]):
                continue
            
            # Skip terminal nodes (e.g., "X0.M1_D")
            if "terminal" in typ:
                continue
            
            feats = []
            
            # Add device type
            if typ in ("n_mosfet", "p_mosfet"):
                feats.append(f"type={typ}")
            elif "res" in typ:
                feats.append("type=resistor")
            elif "cap" in typ:
                feats.append("type=capacitor")
            elif "current" in typ:
                feats.append("type=current_source")
            elif "voltage" in typ:
                feats.append("type=voltage_source")
            
            # Initialize feature list (will be populated with params and OP data below)
            feats_by_id[did] = feats
        
        # Second pass: Add parameters from inst_params
        for did, params in inst_params.items():
            if did not in feats_by_id:
                continue  # Skip if not a device node
            
            # Skip if params is None (from Parquet padding)
            if params is None:
                continue
            
            # Add parameters
            for k, v in params.items():
                if k != 'type' and v is not None:  # Skip None values
                    feats_by_id[did].append(f"{k}={v}")

        # Merge OP data
        for did in list(feats_by_id.keys()):
            if did not in op_data:
                continue
            
            op = op_data[did]
            # Skip if op_data entry is None (from Parquet padding)
            if op is None:
                continue
            
            typ = typ_by_id.get(did, "")

            # MOSFET
            if "mos" in typ or "fet" in typ:
                for k, unit in [('vgs', 'V'), ('vds', 'V'), ('vth', 'V'), ('vdsat', 'V'), ('ids', 'A'), ('gm', 'S')]:
                    val = self._fmt(op.get(k), unit)
                    if val:
                        feats_by_id[did].append(f"{k.capitalize()}={val}")
                # Use the region field of the OP data when present
                region_code = op.get('region')
                if region_code is not None:
                    # Spectre region codes: 0=cutoff, 1=triode, 2=saturation, 3=subthreshold, 4=breakdown
                    region_map = {0: 'cutoff', 1: 'triode', 2: 'saturation', 3: 'subthreshold', 4: 'breakdown'}
                    region_str = region_map.get(int(region_code), 'unknown')
                    feats_by_id[did].append(f"region={region_str}")
                else:
                    # Fallback (no region code in the OP data): derive the region from vgs / vth / vds / vdsat
                    vgs_raw = op.get("vgs")
                    vth_raw = op.get("vth")
                    vds_raw = op.get("vds")
                    vdsat_raw = op.get("vdsat")
                    
                    if all(v is not None for v in [vgs_raw, vth_raw, vds_raw, vdsat_raw]):
                        try:
                            vgsf = abs(float(vgs_raw))
                            vthf = abs(float(vth_raw))
                            vdsf = abs(float(vds_raw))
                            vdsatf = abs(float(vdsat_raw))
                            
                            # Determine region with subthreshold
                            if vgsf <= vthf:
                                if vdsf > 0.1:  # vds > 100mV
                                    region_str = 'subthreshold'
                                else:
                                    region_str = 'cutoff'
                            elif vdsf >= vdsatf:
                                region_str = 'sat'
                            else:
                                region_str = 'triode'
                            
                            feats_by_id[did].append(f"region={region_str}")
                        except:
                            pass
            
            # Resistor
            elif "res" in typ:
                v_val = self._fmt(op.get('v') or op.get('vdrop'), 'V')
                i_val = self._fmt(op.get('i') or op.get('ids'), 'A')
                if v_val: feats_by_id[did].append(f"Vdrop={v_val}")
                if i_val: feats_by_id[did].append(f"I={i_val}")
            
            # Capacitor
            elif "cap" in typ:
                v_val = self._fmt(op.get('v'), 'V')
                i_val = self._fmt(op.get('i'), 'A')
                if v_val: feats_by_id[did].append(f"Vdrop={v_val}")
                if i_val: feats_by_id[did].append(f"I={i_val}")
            
            # Current Source
            elif "current" in typ:
                v_val = self._fmt(op.get('v') or op.get('vdrop'), 'V')
                if v_val: feats_by_id[did].append(f"Vdrop={v_val}")
            
            # Voltage Source
            elif "voltage" in typ:
                i_val = self._fmt(op.get('i') or op.get('ids'), 'A')
                if i_val: feats_by_id[did].append(f"I={i_val}")
        
        # ✅ Emit in canonical order
        def canonical_key(s):
            s_l = s.lower()
            pri = 2
            if s_l in ("vdd","vss","0","gnd"): pri = 0
            elif s_l in ("vin","vout"): pri = 1
            return (pri, s_l)

        # Debug logging if features are empty
        if not feats_by_id:
            print(f"[DEBUG _emit_features] WARNING: feats_by_id is empty!")
            print(f"  nodes count: {len(nodes)}")
            print(f"  inst_params count: {len(inst_params)}")
            print(f"  id2handle count: {len(id2handle)}")
            if nodes:
                print(f"  Sample node IDs: {[n['id'] for n in nodes[:3]]}")
            if id2handle:
                print(f"  Sample handle mappings: {dict(list(id2handle.items())[:3])}")

        emitted_count = 0
        for rid in sorted(feats_by_id.keys(), key=canonical_key):
            if rid in id2handle:
                h = id2handle[rid]
                lines.append(f"{h} id={rid} " + " ".join(feats_by_id[rid]))
                emitted_count += 1

        if emitted_count == 0:
            print(f"[DEBUG _emit_features] WARNING: No features emitted!")
            print(f"  feats_by_id keys ({len(feats_by_id)}): {list(feats_by_id.keys())}")
            print(f"  id2handle keys ({len(id2handle)}): {list(id2handle.keys())[:10]}")

        lines.append("</FEATURES>\n")
        return "\n".join(lines)
    
    def _fmt(self, val, unit):
        """Format value with engineering notation."""
        if val is None or not math.isfinite(val): return None
        ax = abs(val)
        sym, sc = "", 1.0
        for s, scal in (("G",1e9),("M",1e6),("k",1e3),("",1.0),("m",1e-3),("u",1e-6),("n",1e-9),("p",1e-12)):
            if ax >= scal: sym, sc = s, scal; break
        scaled = val / sc
        txt = f"{scaled:.0f}" if abs(scaled) >= 100 else (f"{scaled:.1f}" if abs(scaled) >= 10 else f"{scaled:.2g}")
        return f"{txt} {sym}{unit}"

    def _make_idmaps(self, raw_ids):
        """Create bidirectional ID↔handle mappings."""
        def key(s):
            s_l = s.lower()
            pri = 2
            if s_l in ("vdd","vss","0","gnd"): pri = 0
            elif s_l in ("vin","vout"): pri = 1
            return (pri, s_l)
        ids = sorted(raw_ids, key=key)
        h2id = {f"@{i}": rid for i, rid in enumerate(ids)}
        id2h = {rid: h for h, rid in h2id.items()}
        return id2h, h2id

    def _emit_pinmap(self, id2handle, raw_ids):
        """Generate PINMAP block for MOSFET terminals."""
        suffixes, pins = ("D","G","S","B"), {}
        for rid in raw_ids:
            if "_" in rid:
                base, suf = rid.rsplit("_", 1)
                if suf in suffixes and rid in id2handle:
                    pins.setdefault(base, {})[suf] = id2handle[rid]
        if not pins: return ""
        lines = ["<PINMAP>"]
        for base in sorted(pins.keys()):
            p = pins[base]
            lines.append(f"{base}: " + " ".join([f"{k}={p[k]}" for k in suffixes if k in p]))
        lines.append("</PINMAP>\n")
        return "\n".join(lines)

    def _emit_edges(self, G, id2handle):
        """Generate EDGES block."""
        adj = {}
        for u, v in G.edges():
            # ✅ Use full node names (no splitting)
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        lines = ["<EDGES>"]
        keys = sorted(adj.keys(), key=lambda x: (x not in id2handle, id2handle.get(x, '')))
        for k in keys:
            if k in id2handle:
                nbrs = [id2handle[n] for n in sorted(adj[k]) if n in id2handle]
                if nbrs: lines.append(f"{id2handle[k]}: " + " ".join(nbrs))
        lines.append("</EDGES>\n")
        return "\n".join(lines)
    
    def generate_admissible_actions(self, nodes_data, id2handle, netlist_str=None):
        """
        Generate admissible_actions dict with both xml (handles) and netlist (device_ids) formats.
        
        Excludes:
        - Top-level resistors and capacitors (load/coupling elements)
        - All voltage sources (fixed supply voltages)
        
        Returns:
            dict with 'xml' and 'netlist' formats
        """
        valid = {"n_mosfet","p_mosfet","resistor","capacitor","current_source"}
        
        # Filter valid devices
        device_ids = []
        for n in nodes_data:
            device_type = (n.get('device_type') or "").lower()
            node_id = n['id']
            
            # Check if device type is valid
            if not any(vt in device_type for vt in valid):
                continue
            
            # Exclude top-level resistors and capacitors (R0, C0, etc. without subcircuit prefix)
            if ("resistor" in device_type or "capacitor" in device_type) and '.' not in node_id:
                continue
            
            device_ids.append(node_id)
        
        device_ids = sorted(device_ids)
        
        # Generate handles for xml format
        handles = [id2handle[did] for did in device_ids if did in id2handle]
        
        # Params by type (fallback when target-specific params are unavailable)
        params_by_type = {
            "n_mosfet": ["w", "l"],
            "p_mosfet": ["w", "l"],
            "resistor": ["r"],
            "capacitor": ["c"],
            "current_source": ["dc"]
        }

        # Build target-specific parameter names from netlist:
        # - Prefer macro/shared parameter names (e.g., Mpair_w, Mtail_l)
        # - Fallback to primitive keys (w/l/r/c/dc) when direct numeric values are used.
        params_by_target_ids = {}
        if netlist_str:
            inst_to_allowed = {}
            for ln in netlist_str.splitlines():
                ln = ln.strip()
                if not ln or ln.startswith('//') or ln.startswith('#') or ln.lower().startswith(
                    ('subckt', 'ends', 'parameters', 'simulator', 'global', 'include')
                ):
                    continue
                parts = ln.split()
                if not parts:
                    continue
                inst = parts[0]
                tail = ln.split(')', 1)[-1] if ')' in ln else ln
                kvs = {k.lower(): v for k, v in self.SC_PATTERN.findall(tail)}
                if not kvs:
                    continue
                allowed = []
                for k, v in kvs.items():
                    if k not in ("w", "l", "r", "c", "dc"):
                        continue
                    # A symbolic value (e.g. w_MM3) is itself the admissible parameter, even without a `parameters` line.
                    if re.match(r'^[A-Za-z_]\w*$', v):
                        allowed.append(v)
                    else:
                        allowed.append(k)
                if allowed:
                    inst_to_allowed[inst] = sorted(set(allowed))

            # Map local instance names back to graph node IDs (e.g., X0.M1 -> M1 local line)
            for did in device_ids:
                local = did.split('.')[-1]
                allowed = inst_to_allowed.get(local)
                if allowed:
                    params_by_target_ids[did] = allowed

        # Ensure every target has a parameter list.
        if params_by_target_ids:
            node_type_by_id = {n['id']: (n.get('device_type') or '').lower() for n in nodes_data}
            for did in device_ids:
                if did in params_by_target_ids:
                    continue
                dev_type = node_type_by_id.get(did, '')
                fallback = []
                for t, plist in params_by_type.items():
                    if t in dev_type:
                        fallback = plist
                        break
                if fallback:
                    params_by_target_ids[did] = fallback
            params_by_target_handles = {
                id2handle[did]: params_by_target_ids[did]
                for did in device_ids
                if did in params_by_target_ids and did in id2handle
            }
        else:
            params_by_target_handles = {}
        
        return {
            "xml": {
                "valid_targets": handles,
                "params_by_type": params_by_type,
                "params_by_target": params_by_target_handles,
                "action_format": "increase|decrease <handle> <param_name> <percentage>%",
                "constraints": {
                    "percentage_min": None,
                    "percentage_max": None
                }
            },
            "netlist": {
                "valid_targets": device_ids,
                "params_by_type": params_by_type,
                "params_by_target": params_by_target_ids,
                "action_format": "increase|decrease <device_id> <param_name> <percentage>%",
                "constraints": {
                    "percentage_min": None,
                    "percentage_max": None
                }
            }
        }
