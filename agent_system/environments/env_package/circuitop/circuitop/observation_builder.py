"""
Observation builder for CircuitOp environment.

Converts netlist + simulation results to XML-formatted observations using GraphXMLGenerator.
"""
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add path to graph_serializer
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
data_pre_path = project_root / "circuit" / "data_pre"
if str(data_pre_path) not in sys.path:
    sys.path.insert(0, str(data_pre_path))

from graph_serializer import GraphXMLGenerator


class CircuitObservationBuilder:
    """Thin wrapper around GraphXMLGenerator: static structure once per episode, dynamic state (metrics +
    operating points) each step."""
    
    def __init__(self, circuit_name: str = "circuit", pin_names: Optional[list] = None):
        """circuit_name: topology name; pin_names: e.g. ['vout', 'vin', 'vdd', 'vss', '0']."""
        self.circuit_name = circuit_name
        pin_names = pin_names or ['vout', 'vb_in', 'vin', 'vbg', 'vdd', 'vss', '0']
        self.xml_gen = GraphXMLGenerator(pin_names=pin_names)
        
        # Cache static structure (topology) - generated once per episode
        self.static_xml = None
        self.handle2id = None
        self.id2handle = None
        self.nodes_data = None
        self.graph = None
        self.representation = "xml"
        self.main_instance_name = None
        self.support_instance_names = []
    
    def reset(
        self,
        netlist_text: str,
        static_xml: str = None,
        representation: str = "xml",
        main_instance_name: Optional[str] = None,
        support_instance_names: Optional[list] = None,
    ):
        """Load the episode's static structure from the circuit-only netlist_text (testbench stripped) or the
        optional pre-generated static_xml."""
        self.representation = representation
        self.main_instance_name = main_instance_name
        self.support_instance_names = list(support_instance_names or [])

        # Parse netlist to graph (only circuit devices; testbench not in netlist_text when from env)
        self.graph = self.xml_gen.parse_spectre_to_graph(netlist_text)
        
        if static_xml:
            # Extract handle mappings from static XML and PRESERVE them
            self.static_xml = static_xml
            self._extract_mappings_from_xml(static_xml)
            
            # nodes_data from the parsed graph; the topology is fixed, so node IDs match the static XML
            nodes_data = []
            for n, d in self.graph.nodes(data=True):
                d['id'] = n
                nodes_data.append(d)
            self.nodes_data = nodes_data
        else:
            # Generate graph from netlist (creates handle mappings from scratch)
            self.static_xml, self.handle2id, self.nodes_data = self.xml_gen.generate_static_structure(
                netlist_text, 
                self.circuit_name
            )
            
            # Create reverse mapping
            self.id2handle = {v: k for k, v in self.handle2id.items()}
    
    def _extract_mappings_from_xml(self, xml_text: str):
        """handle2id / id2handle from the static XML's GRAPH_NODE_IDMAP."""
        import re
        
        self.handle2id = {}
        
        # Extract IDMAP section
        idmap_match = re.search(r'<GRAPH_NODE_IDMAP>(.*?)</GRAPH_NODE_IDMAP>', xml_text, re.DOTALL)
        if not idmap_match:
            return
        
        idmap_content = idmap_match.group(1)
        
        # Parse each line: "type=xxx: @0=id0 @1=id1 ..."
        for line in idmap_content.strip().split('\n'):
            if not line.strip() or ':' not in line:
                continue
            
            # Split on first colon
            _, mappings = line.split(':', 1)
            
            # Parse @handle=id pairs
            for pair in mappings.strip().split():
                if '=' in pair:
                    handle, node_id = pair.split('=', 1)
                    self.handle2id[handle] = node_id
        
        # Create reverse mapping
        self.id2handle = {v: k for k, v in self.handle2id.items()}
    
    def _extract_nodes_from_xml(self, xml_text: str):
        """nodes_data ({'id', 'device_type'} dicts) from the static XML's GRAPH_NODE_IDMAP."""
        import re
        
        nodes_data = []
        
        # Extract IDMAP section
        idmap_match = re.search(r'<GRAPH_NODE_IDMAP>(.*?)</GRAPH_NODE_IDMAP>', xml_text, re.DOTALL)
        if not idmap_match:
            return nodes_data
        
        idmap_content = idmap_match.group(1)
        
        # Parse each line: "type=xxx: @0=id0 @1=id1 ..."
        for line in idmap_content.strip().split('\n'):
            if not line.strip() or ':' not in line:
                continue
            
            # Extract type
            type_part, mappings_part = line.split(':', 1)
            device_type = type_part.replace('type=', '').strip()
            
            # Parse @handle=id pairs
            for pair in mappings_part.strip().split():
                if '=' in pair:
                    handle, node_id = pair.split('=', 1)
                    nodes_data.append({
                        'id': node_id,
                        'device_type': device_type
                    })
        
        return nodes_data
    
    def build_observation(
        self,
        netlist_text: str,
        metrics: Dict[str, float],
        op_data: Dict[str, Any],
        step: int,
        constraint_specs=None,
    ) -> str:
        """
        Dynamic-state observation (PERFORMANCE + FEATURES) for this step.

        metrics: 'gain', 'power', 'bandwidth_MHz', ...; op_data: {device_id: op_dict};
        constraint_specs: spec names shown in PERFORMANCE (None = all).
        """
        # Extract instance parameters from netlist
        inst_params = self.xml_gen.scan_instance_params_aligned(netlist_text, self.graph)

        # Debug: Ensure data structures are initialized
        if not self.nodes_data:
            print(f"[ERROR OBS_BUILDER] nodes_data is None or empty! Cannot build observation.")
            print(f"  id2handle exists: {self.id2handle is not None}")
            print(f"  graph exists: {self.graph is not None}")
            return "</PERFORMANCE>\n<FEATURES>\n</FEATURES>"

        if not self.id2handle:
            print(f"[ERROR OBS_BUILDER] id2handle is None or empty! Cannot build observation.")
            print(f"  nodes_data exists: {self.nodes_data is not None}")
            return "</PERFORMANCE>\n<FEATURES>\n</FEATURES>"

        # Generate dynamic state (performance + features)
        dynamic_xml = self.xml_gen.generate_dynamic_state(
            metrics=metrics,
            nodes=self.nodes_data,
            inst_params=inst_params,
            op_data=op_data,
            id2handle=self.id2handle,
            constraint_specs=constraint_specs,
        )

        if self.representation == "netlist":
            return self._build_netlist_observation(netlist_text, dynamic_xml)

        return dynamic_xml

    @staticmethod
    def _extract_tagged_block(text: str, tag_name: str) -> str:
        match = re.search(rf"(<{tag_name}>.*?</{tag_name}>)", text, re.DOTALL)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_feature_map(dynamic_xml: str) -> Dict[str, str]:
        feature_match = re.search(r"<FEATURES>(.*?)</FEATURES>", dynamic_xml, re.DOTALL)
        if not feature_match:
            return {}

        feature_map: Dict[str, str] = {}
        for raw_line in feature_match.group(1).splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = re.match(r"^\S+\s+id=(\S+)\s+(.*)$", line)
            if not match:
                continue

            feature_text = match.group(2).strip()
            feature_text = re.sub(r"(^|\s)type=\S+\s*", " ", feature_text).strip()
            feature_text = re.sub(r"\s{2,}", " ", feature_text)
            feature_map[match.group(1)] = feature_text

        return feature_map

    @staticmethod
    def _extract_op_feature_map(feature_map: Dict[str, str]) -> Dict[str, str]:
        op_keys = {"Vgs", "Vds", "Vth", "Vdsat", "Ids", "Gm", "region", "Vdrop", "I"}
        op_feature_map: Dict[str, str] = {}
        for device_id, feature_text in feature_map.items():
            op_tokens = []
            for token in feature_text.split():
                key = token.split("=", 1)[0]
                if key in op_keys:
                    op_tokens.append(token)
            if op_tokens:
                op_feature_map[device_id] = " ".join(op_tokens)
        return op_feature_map

    @staticmethod
    def _parse_top_level_instances(netlist_text: str) -> Dict[str, list[str]]:
        subckt_to_instances: Dict[str, list[str]] = {}
        in_subckt = False

        for raw_line in netlist_text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue
            lowered = stripped.lower()
            if lowered.startswith("subckt"):
                in_subckt = True
                continue
            if lowered.startswith("ends"):
                in_subckt = False
                continue
            if in_subckt or not stripped.startswith("X"):
                continue

            match = re.search(r"\)\s+(\S+)", stripped)
            if not match:
                continue
            subckt_name = match.group(1)
            inst_name = stripped.split()[0]
            subckt_to_instances.setdefault(subckt_name, []).append(inst_name)

        return subckt_to_instances

    def _instance_role(self, inst_name: str) -> Optional[str]:
        if inst_name == self.main_instance_name:
            return "main_body"
        if inst_name in self.support_instance_names:
            return "bias_block"
        return None

    def _annotate_netlist(self, netlist_text: str, feature_map: Dict[str, str]) -> str:
        subckt_to_instances = self._parse_top_level_instances(netlist_text)
        current_subckt: Optional[str] = None
        annotated_lines = []

        for raw_line in netlist_text.splitlines():
            stripped = raw_line.strip()
            lowered = stripped.lower()

            if not stripped:
                annotated_lines.append(raw_line)
                continue

            if lowered.startswith("parameters"):
                annotated_lines.append(f"{raw_line} // current editable parameter values")
                continue

            if lowered.startswith("subckt"):
                parts = stripped.split()
                current_subckt = parts[1] if len(parts) >= 2 else None
                parent_instances = subckt_to_instances.get(current_subckt or "", [])
                if parent_instances:
                    rendered = ", ".join(
                        f"{name}({self._instance_role(name) or 'subckt_instance'})"
                        for name in parent_instances
                    )
                    annotated_lines.append(f"{raw_line} // instantiated_by={rendered}")
                else:
                    annotated_lines.append(raw_line)
                continue

            if lowered.startswith("ends"):
                current_subckt = None
                annotated_lines.append(raw_line)
                continue

            if stripped.startswith("//") or stripped.startswith("#"):
                annotated_lines.append(raw_line)
                continue

            token = stripped.split()[0]
            comment = None

            if current_subckt:
                parent_instances = subckt_to_instances.get(current_subckt, [])
                hierarchical_ids = [
                    f"{parent}.{token}"
                    for parent in parent_instances
                    if f"{parent}.{token}" in feature_map
                ]
                if hierarchical_ids:
                    comment = " | ".join(
                        f"{hier_id}: {feature_map[hier_id]}"
                        for hier_id in hierarchical_ids
                    )
            else:
                if token.startswith("X"):
                    role = self._instance_role(token)
                    match = re.search(r"\)\s+(\S+)", stripped)
                    subckt_name = match.group(1) if match else None
                    comment_parts = []
                    if role:
                        comment_parts.append(f"role={role}")
                    if subckt_name:
                        comment_parts.append(f"subckt={subckt_name}")
                    if comment_parts:
                        comment = " ".join(comment_parts)
                elif token in feature_map:
                    comment = f"{token}: {feature_map[token]}"

            annotated_lines.append(f"{raw_line} // {comment}" if comment else raw_line)

        return "\n".join(annotated_lines).strip()

    def _build_netlist_observation(self, netlist_text: str, dynamic_xml: str) -> str:
        performance_block = self._extract_tagged_block(dynamic_xml, "PERFORMANCE")
        feature_map = self._extract_feature_map(dynamic_xml)
        op_feature_map = self._extract_op_feature_map(feature_map)
        annotated_netlist = self._annotate_netlist(netlist_text, op_feature_map)
        return f"{performance_block}\n<FEATURES>\n{annotated_netlist}\n</FEATURES>\n"
    
