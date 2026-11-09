"""
State utilities for CircuitOp environment.

Helper functions only; observation building is handled by
observation_builder.py using GraphXMLGenerator.
"""
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from .action_validator import ActionValidator

sys.path.insert(0, str(Path(__file__).parents[5] / "circuit" / "data_pre"))
from netlist_gen import (  # noqa: E402  (single copies of the technology / netlist helpers)
    TECH_MODEL_MAP, TECH_ROOT, format_si, get_tech_node, prepare_template_bundle, resolve_tech_model,
    split_circuit_and_testbench, wrap_netlist,
)
def _resolve_circuit_kind_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    metadata = metadata or {}
    circuit_kind = metadata.get("circuit_kind")
    if circuit_kind:
        return str(circuit_kind).strip().lower()

    complexity = str(metadata.get("complexity", "")).strip().lower()
    signal_mode = str(metadata.get("signal_mode", "")).strip().lower()
    if complexity == "complex":
        if signal_mode == "fully_differential":
            return "complex_fully_differential"
        if signal_mode == "single_ended":
            return "complex_single_ended"
    if complexity == "simple":
        return "simple"
    return None







# Per-node L_min overrides (from simulator convergence probing); other nodes use 45 nm up to 65 nm,
# else the node size.
_L_MIN_OVERRIDE_NM = {
    "22nm":  22,   # use node value; 45/65nm share the 45nm floor below
    "90nm":  56,   # actual Spectre limit ~55.5nm
    "130nm": 85,   # actual Spectre limit ~84.5nm
}
_L_MIN_SMALL_NODE_NM = 45   # floor for nodes 32nm–65nm

def _tech_node_to_length_min(node: str) -> float:
    if node in _L_MIN_OVERRIDE_NM:
        return _L_MIN_OVERRIDE_NM[node] * 1e-9
    match = re.fullmatch(r"(\d+)nm", str(node).strip())
    if not match:
        return 30e-9
    node_nm = float(match.group(1))
    if node_nm <= 65:
        return _L_MIN_SMALL_NODE_NM * 1e-9
    return node_nm * 1e-9


TECH_GEOMETRY_BOUNDS = {
    node: {
        "l": (_tech_node_to_length_min(node), 100e-6),
        "w": (45e-9, 500e-6),
    }
    for node in TECH_MODEL_MAP
}
# per node: L from the node's minimum length to 100 µm, W from 45 nm to 500 µm

# 65nm per-topology L upper caps (µm); unlisted topologies keep the 100 µm ceiling. Beyond a cap the topology
# develops an in-band shelf at typical operating points; the caps aid search, the in-band flatness gate stays
# the correctness check.
TOPOLOGY_L_MAX_UM_65NM = {
    "common_gate_amplifier_with_resistive_load": 2.5,
    "common_gate_amplifier_with_resistive_load_pmos_input": 2.5,
    "common_source_amplifier_with_resistive_load_pmos_input": 2.5,
    "common_source_amplifier_with_resistive_load": 6.0,
    "common_source_amplifier_with_diode_connected_nmos_load_pmos_input": 6.0,
    "common_source_resistive_degeneration_with_resistive_load": 15.0,
    "common_source_amplifier_with_current_source_load_pmos_input": 40.0,
    "common_source_resistive_source_degeneration_with_current_source_load": 40.0,
    "common_source_resistive_source_degeneration_with_current_source_load_pmos_input": 40.0,
}


def get_tech_geometry_bounds(node: Optional[str] = None, topology: Optional[str] = None) -> Dict[str, tuple]:
    """Tech-wide geometry bounds; with `topology` given, the 65nm per-topology
    L cap (TOPOLOGY_L_MAX_UM_65NM) tightens the L upper bound.
    Non-65nm nodes ignore the cap (the table was measured on 65nm only)."""
    tech_node = node or get_tech_node()
    if tech_node not in TECH_GEOMETRY_BOUNDS:
        raise RuntimeError(f"no geometry bounds for technology node {tech_node!r}; choices: {sorted(TECH_GEOMETRY_BOUNDS)}")
    bounds = TECH_GEOMETRY_BOUNDS[tech_node]
    if topology and tech_node == "65nm":
        cap_um = TOPOLOGY_L_MAX_UM_65NM.get(str(topology))
        if cap_um is not None:
            l_min, l_max = bounds["l"]
            bounds = {**bounds, "l": (l_min, min(l_max, cap_um * 1e-6))}
    return bounds


def format_geometry(x: float) -> str:
    """Geometry length in nm / um / mm for bounds messages (a bare SPICE "1.9m" reads as metres);
    netlists and FEATURES keep SPICE suffixes."""
    for unit, factor in (("mm", 1e-3), ("um", 1e-6), ("nm", 1e-9)):
        if abs(x) >= factor:
            return f"{round(x / factor * 10.0) / 10.0:g}{unit}"
    return f"{x:g}m"


def format_tech_geometry_bounds(node: Optional[str] = None, topology: Optional[str] = None) -> str:
    tech_node = node or get_tech_node()
    bounds = get_tech_geometry_bounds(tech_node, topology=topology)
    l_min, l_max = bounds["l"]
    w_min, w_max = bounds["w"]
    capped = (topology and tech_node == "65nm"
              and str(topology) in TOPOLOGY_L_MAX_UM_65NM)
    l_scope = "for this topology " if capped else ""
    return (
        f"For tech {tech_node}, length l must stay within [{format_geometry(l_min)}, {format_geometry(l_max)}] "
        f"{l_scope}and width w must stay within [{format_geometry(w_min)}, {format_geometry(w_max)}]. "
        "Designs outside this range are invalid."
    )


def parse_si(s) -> float:
    """Parse SI notation string to float (e.g., '2.46u' -> 2.46e-6). The FULL string must be a
    number (scientific notation allowed) plus an optional known suffix, so '7.275e+09u'
    is 7275 and '5meg' is 5e6; anything else raises instead of returning a silently-wrong value.
    """
    UNIT_MAP = {'': 1, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9,
                'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9}

    if isinstance(s, (int, float)):
        return float(s)

    m = re.fullmatch(
        r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(meg|[fpnumkg]?)',
        str(s).strip().lower(),
    )
    if not m:
        raise ValueError(f"Cannot parse SI from '{s}'")
    val, unit = m.groups()
    return float(val) * UNIT_MAP[unit]


def get_netlist_for_observation(full_netlist: str) -> str:
    """
    Return netlist with testbench removed, for building observation only.
    Simulation still uses the full netlist; the agent sees only the circuit part.
    """
    circuit_only, _ = split_circuit_and_testbench(full_netlist)
    if not circuit_only:
        return full_netlist
    # Preserve parameters line so graph parser can resolve param values
    for line in full_netlist.splitlines():
        s = line.strip()
        if s.lower().startswith("parameters "):
            return s + "\n" + circuit_only
    return circuit_only


def _extract_parameter_lines(netlist_text: str) -> List[str]:
    return [
        line.strip()
        for line in netlist_text.splitlines()
        if line.strip().lower().startswith("parameters ")
    ]


_ENOT_SI_RE = re.compile(r'^([+-]?\d+\.?\d*[eE][+-]?\d+)([fpnumkMG])$')
_ENOT_SI_MULT = {'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
                 'k': 1e3, 'M': 1e6, 'G': 1e9}


def _spectre_safe_value(v):
    """Fold e-notation + SI-suffix hybrids that Spectre rejects (e.g. ``2.706e+09u`` = 2706) into a plain
    number; other values pass through unchanged."""
    s = str(v).strip()
    m = _ENOT_SI_RE.match(s)
    if not m:
        return v
    return f"{float(m.group(1)) * _ENOT_SI_MULT[m.group(2)]:.6g}"


def _extract_parameter_map(netlist_text: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for line in netlist_text.splitlines():
        match = _PARAMETERS_LINE_RE.match(line)
        if not match:
            continue
        for key, value in _PARAM_PAIR_RE.findall(match.group(1)):
            params[key] = value
    return params


def construct_observation_netlist(
    task_data: Dict[str, Any],
    netlist_text: Optional[str] = None,
) -> str:
    """
    Observation netlist: dataset structure plus the current parameters line, so complex topologies show the
    canonical X0/X1 structure rather than the duplicated testbench instances (XDUT_ADM, ...).
    """
    structure = task_data.get('structure', {}) if task_data else {}
    structure_netlist = structure.get('netlist', '')
    if not structure_netlist:
        return get_netlist_for_observation(netlist_text or '')

    editable_params = task_data.get('initial_state', {}).get('params', {}) if task_data else {}
    filtered_params = {k: v for k, v in editable_params.items() if v is not None}

    param_lines: List[str] = []
    current_param_map = _extract_parameter_map(netlist_text or '')
    if filtered_params:
        rendered_pairs = []
        for key, fallback_value in filtered_params.items():
            rendered_pairs.append(
                f"{key}={_spectre_safe_value(current_param_map.get(key, fallback_value))}")
        param_lines = ["parameters " + " ".join(rendered_pairs)]
    else:
        param_lines = _extract_parameter_lines(netlist_text or '')

    blocks = []
    if param_lines:
        blocks.extend(param_lines)
    blocks.append(structure_netlist.strip())
    return "\n".join(blocks).strip()


_PARAMETERS_LINE_RE = re.compile(r"^\s*parameters\b(.*)$", re.IGNORECASE)
_PARAM_PAIR_RE = re.compile(r"([A-Za-z_]\w*)=([^\s]+)")
_INSTANCE_GEOMETRY_RE = re.compile(r"\b([WwLl])=([^\s]+)")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def find_geometry_bound_violations(netlist_text: str, node: Optional[str] = None,
                                   topology: Optional[str] = None) -> List[str]:
    bounds = get_tech_geometry_bounds(node, topology=topology)
    params: Dict[str, str] = {}
    violations: List[str] = []
    seen = set()

    for line in netlist_text.splitlines():
        match = _PARAMETERS_LINE_RE.match(line)
        if not match:
            continue
        for key, value in _PARAM_PAIR_RE.findall(match.group(1)):
            params[key] = value

    for line in netlist_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith((
            "parameters",
            "simulator",
            "global",
            "include",
            "subckt",
            "ends",
            "saveoptions",
            "simulatoroptions",
            "dcop",
            "dcopinfo",
            "ac ",
        )):
            continue

        inst_name = stripped.split()[0]
        if not inst_name.upper().startswith("M"):
            continue
        for key, raw_value in _INSTANCE_GEOMETRY_RE.findall(line):
            geom_key = key.lower()
            resolved_value = raw_value
            if _IDENTIFIER_RE.fullmatch(raw_value):
                resolved_value = params.get(raw_value, raw_value)
            try:
                numeric_value = parse_si(resolved_value)
            except Exception:
                continue

            lower, upper = bounds[geom_key]
            # Relative tolerance: SI-string parsing and decimal bound literals differ by ULPs at the boundary
            eps = 1e-9
            if lower * (1 - eps) <= numeric_value <= upper * (1 + eps):
                continue

            value_str = format_geometry(numeric_value)
            value_display = (
                f"{raw_value}->{value_str}" if raw_value != resolved_value else value_str
            )
            violation = f"{geom_key}={value_display} outside [{format_geometry(lower)}, {format_geometry(upper)}]"
            if violation not in seen:
                seen.add(violation)
                violations.append(violation)

    return violations


def find_near_bound_params(netlist_text: str, node: Optional[str] = None,
                           topology: Optional[str] = None,
                           factor: float = 2.0) -> Dict[str, List[str]]:
    """Geometry params one typical action away from a bound: near_floor (value < lower * factor) and
    near_cap (value > upper / factor), e.g. {"near_floor": ["l_MM12=60n"], "near_cap": []}.
    """
    bounds = get_tech_geometry_bounds(node, topology=topology)
    params: Dict[str, str] = {}
    for line in netlist_text.splitlines():
        match = _PARAMETERS_LINE_RE.match(line)
        if match:
            for key, value in _PARAM_PAIR_RE.findall(match.group(1)):
                params[key] = value
    out = {"near_floor": [], "near_cap": []}
    seen = set()
    for line in netlist_text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.split()[0].upper().startswith("M"):
            continue
        for key, raw_value in _INSTANCE_GEOMETRY_RE.findall(line):
            geom_key = key.lower()
            resolved = params.get(raw_value, raw_value) if _IDENTIFIER_RE.fullmatch(raw_value) else raw_value
            try:
                v = parse_si(resolved)
            except Exception:
                continue
            lower, upper = bounds[geom_key]
            name = raw_value if _IDENTIFIER_RE.fullmatch(raw_value) else geom_key
            tag = f"{name}={format_si(v)}"
            if tag in seen:
                continue
            if lower <= v < lower * factor:
                seen.add(tag)
                out["near_floor"].append(tag)
            elif upper / factor < v <= upper:
                seen.add(tag)
                out["near_cap"].append(tag)
    return out


def construct_netlist_from_dataset(
    task_data: Dict[str, Any],
    representation: str = 'netlist',
) -> str:
    """
    Complete simulation netlist from the dataset structure and params, with the testbench generated by
    netlist_gen (the dataset's structure.testbench is not used).
    """
    simulation_bundle = build_simulation_bundle_from_dataset(task_data)
    full_body = (
        simulation_bundle["simulation_body"].strip()
        + "\n"
        + simulation_bundle["testbench"].strip()
    ).strip()
    return wrap_netlist(
        full_body,
        simulation_bundle["params"],
        analysis_kind=simulation_bundle.get("kind"),
    )


def build_simulation_bundle_from_dataset(
    task_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Rebuild the simulation body and testbench from dataset structure using netlist_gen.

    The dataset only needs to provide:
    - structure.netlist
    - metadata.topology_name
    - initial_state.params
    """
    structure = task_data.get('structure', {})
    core_netlist = structure.get('netlist', '')
    if not core_netlist:
        raise ValueError("No netlist found in task_data['structure']['netlist']")
    topology_name = task_data.get('metadata', {}).get('topology_name')
    if not topology_name:
        raise ValueError("No topology_name found in task_data['metadata']['topology_name']")

    params = task_data.get('initial_state', {}).get('params', {})
    if not params:
        raise ValueError("No params found in task_data['initial_state']['params']")

    metadata = task_data.get('metadata', {}) or {}
    bundle = prepare_template_bundle(
        topology_name,
        core_netlist,
        kind_override=_resolve_circuit_kind_from_metadata(metadata),
    )
    filtered_params = {key: val for key, val in params.items() if val is not None}
    return {
        "topology_name": topology_name,
        "kind": bundle.get("kind"),
        "main_subckt_name": bundle.get("main_subckt_name"),
        "main_instance_name": bundle.get("main_instance_name"),
        "support_instance_names": bundle.get("support_instance_names", []),
        "simulation_body": bundle["simulation_body"],
        "testbench": bundle["testbench"],
        "params": filtered_params,
    }


def get_admissible_actions(validator: Optional[ActionValidator] = None) -> List[str]:
    """Admissible-action lines from the validator, or generic descriptions when no targets are loaded."""
    if validator and validator._targets_loaded:
        # Return specific actions from TARGETS section
        return validator.get_admissible_actions_text()
    
    # Fallback: generic action descriptions
    admissible = [
        "Adjust MOSFET width (w parameter)",
        "Adjust MOSFET length (l parameter)",
        "Adjust resistor value (r parameter)",
        "Adjust capacitor value (c parameter)",
        "Adjust current source DC value"
    ]
    
    return admissible
