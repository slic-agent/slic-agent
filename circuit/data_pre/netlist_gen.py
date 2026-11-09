#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- SI UNIT UTILITIES ---
UNIT_MAP = {'': 1, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9,
            'u': 1e-6, 'm': 1e-3, 'k': 1e3, 'meg': 1e6, 'g': 1e9}

def parse_si(s):
    """Parse SI token or plain/scientific number to float.

    The number is read first ("1.3e-06" stays 1.3e-06), then a strict full-string SI match;
    anything else raises instead of returning a wrong mantissa.
    """
    if isinstance(s, (int, float, np.number)):
        return float(s)
    t = str(s).strip().lower()
    try:
        return float(t)
    except ValueError:
        pass
    m = re.fullmatch(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)(meg|[fpnumkg])', t)
    if not m:
        raise ValueError(f"Cannot parse SI from '{s}'")
    val, unit = m.groups()
    return float(val) * UNIT_MAP[unit]


def format_si(x):
    """
    Quantize mantissa to 0.1 resolution with SI suffix.
    """
    x = round(x, 11) # help float precision
    for u, f in reversed(sorted(UNIT_MAP.items(), key=lambda kv: kv[1])):
        if abs(x) >= f:
            mantissa = x / f
            # Quantize mantissa to 0.1 steps (…, -0.2, -0.1, 0.0, 0.1, 0.2, …)
            mantissa_q = round(mantissa * 10.0) / 10.0
            return f"{mantissa_q:g}{u}"
    # Fallback for very small values: quantize directly in base units
    mantissa_q = round(x * 10.0) / 10.0
    return f"{mantissa_q:g}"

# --- TECH MODEL SELECTION (the only copy: circuitop/state_utils.py imports these) ---
TECH_MODEL_MAP = {   # predictive technology model cards under TECH_ROOT: `slic data --cards` (the 65 nm card ships with the repository)
    "22nm":  "22nm_bulk_v2.scs",
    "32nm":  "32nm_HP.scs",     # PTM 2008 High-Performance MG/HK card
    "45nm":  "45nm_HP.scs",     # PTM 2008 High-Performance MG/HK card
    "65nm":  "65nm_bulk.scs",
    "90nm":  "90nm_bulk.scs",
    "130nm": "130nm_bulk.scs",
}
TECH_ROOT = os.environ.get(
    "CIRCUITOP_TECH_ROOT",
    str(Path(__file__).parents[2] / "circuit" / "dataset"),
)


def get_tech_node() -> str:
    """Technology node from CIRCUITOP_TECH_NODE (e.g. 65nm); raises when unset, since it selects the model card and
    geometry box. Training/eval set it from env.circuitop.tech_node."""
    node = os.environ.get("CIRCUITOP_TECH_NODE")
    if not node:
        raise RuntimeError("CIRCUITOP_TECH_NODE is not set; declare the technology node explicitly, e.g. "
                           f"export CIRCUITOP_TECH_NODE=65nm. Choices: {sorted(TECH_MODEL_MAP)}")
    return node


def resolve_tech_model() -> str:
    """The device model card to include: CIRCUITOP_TECH_MODEL (a path) if set, else TECH_MODEL_MAP[CIRCUITOP_TECH_NODE] under
    TECH_ROOT. An unset or unknown node raises."""
    override = os.environ.get("CIRCUITOP_TECH_MODEL")
    if override:
        return override
    node = get_tech_node()
    fname = TECH_MODEL_MAP.get(node)
    if fname is None:
        raise RuntimeError(f"Unknown technology node CIRCUITOP_TECH_NODE={node!r}; choices: {sorted(TECH_MODEL_MAP)}")
    return f"{TECH_ROOT}/{fname}"


# --- NETLIST BLOCKS ---
spectre_header = """simulator lang=spectre
global 0
parameters {parameters_line}
include "{tech_model}" section=tt
"""
spectre_footer = """
simulatorOptions options psfversion="1.4.0" reltol=1e-3 vabstol=1e-6 iabstol=1e-12 temp=27 tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 maxnotes=5 maxwarns=5 digits=5 cols=80 pivrel=1e-3 sensfile="../psf/sens.output" checklimitdest=psf
dcOp dc write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status
dcOpInfo info what=oppoint where=rawfile
ac ac start=1 stop=500G dec=100 annotate=status
saveOptions options save=allpub
"""
# AC sweep from 1 Hz, below every topology's dominant pole: the first point is the reference for the PM phase, the
# -3 dB bandwidth and the in-band flatness. Gain is sampled at 1 kHz (_sample_gain_db).

# Testbenches by topology, appended to the template body for simulation.
# Common-source / cascode style: a 1T inductor from output to input gate sets the DC point by unity feedback and is an
# AC open; a 1T capacitor couples the stimulus. The measured AC transfer is the open-loop gain.
SIMPLE_CIRCUIT_TESTBENCH = """// CIRCUITOP_POWER_SOURCES: V0
LFB (vout vb_in) inductor l=1T
CIN (vb_in vac_in) capacitor c=1T
V1 (vac_in 0) vsource dc=vdd/2 type=dc mag=1
V0 (vdd 0) vsource dc=vdd type=dc
C1 (vout 0) capacitor c=100f"""

# Common-gate: the AC source drives the source node vin through C0 = 1T, so no high-pass corner from coupling into the
# low-impedance source appears above the 1 Hz sweep start.
COMMON_GATE_TESTBENCH = """// CIRCUITOP_POWER_SOURCES: V0
V1 (vac_in 0) vsource dc=0 type=dc mag=1
V0 (vdd 0) vsource dc=vdd type=dc
C0 (vac_in vin) capacitor c=1T
C1 (vout 0) capacitor c=100f"""

def get_testbench_for_topology(topo_name: str) -> str:
    """Return the testbench netlist fragment for the given topology (main-body-only templates)."""
    topo_lower = (topo_name or "").lower()
    if topo_lower.startswith("common_gate_amplifier_"):
        return COMMON_GATE_TESTBENCH
    return SIMPLE_CIRCUIT_TESTBENCH


def parse_subckt_signatures(content: str) -> List[Tuple[str, List[str]]]:
    signatures = []
    for line in content.splitlines():
        s = line.strip()
        if not s.lower().startswith("subckt "):
            continue
        parts = s.split()
        if len(parts) >= 2:
            signatures.append((parts[1], parts[2:]))
    return signatures


def get_subckt_signature(content: str, subckt_name: str) -> Optional[Tuple[str, List[str]]]:
    subckt_lower = subckt_name.lower()
    for name, pins in parse_subckt_signatures(content):
        if name.lower() == subckt_lower:
            return name, pins
    return None


def extract_subckt_block(text: str, subckt_name: str) -> str:
    lines = text.splitlines()
    out = []
    in_block = False
    target = subckt_name.lower()
    for line in lines:
        s = line.strip()
        if s.lower().startswith("subckt "):
            parts = s.split()
            in_block = len(parts) >= 2 and parts[1].lower() == target
        if in_block:
            out.append(line)
        if in_block and s.lower().startswith("ends"):
            break
    return "\n".join(out).strip()


def parse_top_level_instances(content: str) -> List[Dict[str, object]]:
    lines = content.strip().splitlines()
    in_subckt = False
    instances = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            continue
        if s.lower().startswith("subckt "):
            in_subckt = True
            continue
        if s.lower().startswith("ends"):
            in_subckt = False
            continue
        if in_subckt or not s.upper().startswith("X"):
            continue
        match = re.match(r'^(X\w+)\s*\(([^)]*)\)\s+(\w+)(.*)$', s)
        if not match:
            continue
        instances.append({
            "name": match.group(1),
            "nets": match.group(2).split(),
            "subckt": match.group(3),
            "tail": match.group(4).strip(),
            "raw": s,
        })
    return instances


def infer_template_kind(topo_name: str) -> str:
    """Test-bench kind from the topology name: complex OTA (fully differential / single ended) or simple."""
    topo_lower = topo_name.lower()
    if "fully_differential" in topo_lower:
        return "complex_fully_differential"
    if "single_ended" in topo_lower:
        return "complex_single_ended"
    return "simple"


def rewrite_instance_line_with_pins(
    instance: Dict[str, object],
    pin_names: List[str],
    net_map: Dict[str, str],
    main_pin_net_map: Dict[str, str],
    suffix: str = "",
) -> str:
    nets = []
    for idx, net in enumerate(instance["nets"]):
        pin_name = pin_names[idx] if idx < len(pin_names) else None
        if pin_name and pin_name in main_pin_net_map:
            nets.append(net_map.get(pin_name, net_map.get(main_pin_net_map[pin_name], net)))
        else:
            nets.append(net_map.get(net, net))
    name = f'{instance["name"]}{suffix}' if suffix else instance["name"]
    tail = f' {instance["tail"]}' if instance["tail"] else ""
    return f'{name} ({" ".join(nets)}) {instance["subckt"]}{tail}'


def build_complex_ota_testbench(
    topo_name: str,
    main_subckt_name: str,
    main_pins: List[str],
    main_inst_nets: List[str],
    support_instances: List[Dict[str, object]],
    signature_map: Dict[str, List[str]],
    kind: str,
) -> str:
    bias_nets = sorted({
        net for net in list(main_inst_nets) + [n for inst in support_instances for n in inst["nets"]]
        if isinstance(net, str) and net.startswith("LG_")
    })
    main_pin_net_map = {
        pin: net for pin, net in zip(main_pins, main_inst_nets)
        if isinstance(net, str) and net.startswith("LG_")
    }

    def path_net_map(path_name: str) -> Dict[str, str]:
        suffix = path_name.lower()
        net_map = {net: f"{net}_{suffix}" for net in bias_nets}
        for pin, net in main_pin_net_map.items():
            net_map[pin] = net_map.get(net, net)
        if kind == "complex_single_ended":
            output_pins = [p for p in main_pins if p.lower().startswith("vout")]
            output_pin = output_pins[0] if output_pins else "Voutp"
            if path_name == "ADM":
                net_map.update({
                    "Vinn": "vfb_adm",
                    "Vinp": "opin",
                    output_pin: "vout_adm",
                    "vdd": "vdd_adm",
                    "vss": "vss_adm",
                })
            elif path_name == "ACM":
                net_map.update({
                    "Vinn": "vfb_cm",
                    "Vinp": "opin_cm",
                    output_pin: "vout_cm",
                    "vdd": "vdd_acm",
                    "vss": "vss_acm",
                })
            elif path_name == "PSRRP":
                net_map.update({
                    "Vinn": "vout_psrrp",
                    "Vinp": "opin_psrr",
                    output_pin: "vout_psrrp",
                    "vdd": "vdd_psrrp",
                    "vss": "vss_psrrp",
                })
            elif path_name == "PSRRN":
                net_map.update({
                    "Vinn": "vout_psrrn",
                    "Vinp": "opin_psrr",
                    output_pin: "vout_psrrn",
                    "vdd": "vdd_psrrn",
                    "vss": "vss_psrrn",
                })
        else:
            output_pins = [p for p in main_pins if p.lower().startswith("vout")]
            out_n = output_pins[0] if len(output_pins) >= 1 else "Voutn"
            out_p = output_pins[1] if len(output_pins) >= 2 else "Voutp"
            if path_name == "ADM":
                net_map.update({
                    "Vinn": "vinn_fb",
                    "Vinp": "vinp_fb",
                    out_n: "voutn_adm",
                    out_p: "voutp_adm",
                    "vdd": "vdd_adm",
                    "vss": "vss_adm",
                })
            elif path_name == "ACM":
                net_map.update({
                    "Vinn": "vinn_fb_cm",
                    "Vinp": "vinp_fb_cm",
                    out_n: "voutn_cm",
                    out_p: "voutp_cm",
                    "vdd": "vdd_acm",
                    "vss": "vss_acm",
                })
            elif path_name == "PSRRP":
                net_map.update({
                    "Vinn": "voutp_psrrp",
                    "Vinp": "voutn_psrrp",
                    out_n: "voutn_psrrp",
                    out_p: "voutp_psrrp",
                    "vdd": "vdd_psrrp",
                    "vss": "vss_psrrp",
                })
            elif path_name == "PSRRN":
                net_map.update({
                    "Vinn": "voutp_psrrn",
                    "Vinp": "voutn_psrrn",
                    out_n: "voutn_psrrn",
                    out_p: "voutp_psrrn",
                    "vdd": "vdd_psrrn",
                    "vss": "vss_psrrn",
                })
        return net_map

    lines = [
        "parameters supply_voltage=vdd",
        "parameters vcm_ratio=0.5",
        "parameters vcm=supply_voltage*vcm_ratio",
        "parameters vdiff_dc=0",
        "parameters vdiff_ac=1",
        "parameters cload=100f",
        "",
        "// CIRCUITOP_POWER_SOURCES: VDD_ADM VSS_ADM",
        "// one supply pair per testbench path; the reported power is the ADM pair only (one amplifier + its bias block)",
        "VDD_ADM (vdd_adm 0) vsource dc=supply_voltage type=dc",
        "VSS_ADM (vss_adm 0) vsource dc=0 type=dc",
        "VDD_ACM (vdd_acm 0) vsource dc=supply_voltage type=dc",
        "VSS_ACM (vss_acm 0) vsource dc=0 type=dc",
        "VDD_DC (vdd_dc 0) vsource dc=supply_voltage type=dc",
        "VSS_DC (vss_dc 0) vsource dc=0 type=dc",
        "VDD_PSRRP (vdd_psrrp 0) vsource dc=supply_voltage mag=1 type=dc",
        "VSS_PSRRP (vss_psrrp 0) vsource dc=0 type=dc",
        "VDD_PSRRN (vdd_psrrn 0) vsource dc=supply_voltage type=dc",
        "VSS_PSRRN (vss_psrrn 0) vsource dc=0 mag=1 type=dc",
        "",
    ]

    if kind == "complex_single_ended":
        lines.extend([
            "// ADM path",
            "VDC_OPIN (opin 0) vsource dc=vcm type=dc",
            "VIN_ADM (signal_in 0) vsource dc=vcm mag=1 type=dc",
            "LFB_ADM (vout_adm vfb_adm) inductor l=1T",
            "CIN_ADM (vfb_adm signal_in) capacitor c=1T",
        ])
        adm_map = path_net_map("ADM")
        lines.append(
            f'XDUT_ADM ({" ".join(adm_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    adm_map,
                    main_pin_net_map,
                    suffix=f"_ADM_{idx}",
                )
            )
        lines.extend([
            "CLOAD_ADM (vout_adm 0) capacitor c=cload",
            "",
            "// ACM path",
            "VDC_OPIN_CM (opin_cm 0) vsource dc=vcm mag=1 type=dc",
            "VIN_ACM (signal_in_cm 0) vsource dc=vcm mag=1 type=dc",
            "LFB_ACM (vout_cm vfb_cm) inductor l=1T",
            "CIN_ACM (vfb_cm signal_in_cm) capacitor c=1T",
        ])
        acm_map = path_net_map("ACM")
        lines.append(
            f'XDUT_ACM ({" ".join(acm_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    acm_map,
                    main_pin_net_map,
                    suffix=f"_ACM_{idx}",
                )
            )
        lines.extend([
            "CLOAD_ACM (vout_cm 0) capacitor c=cload",
            "",
            "// DC offset path",
            "VDC_OFFSET (opin_dc 0) vsource dc=vcm type=dc",
            "VIN_DC (signal_in_dc 0) vsource dc=vcm mag=0 type=dc",
            "LFB_DC (vout_dc vfb_dc) inductor l=1T",
            "CIN_DC (vfb_dc signal_in_dc) capacitor c=1T",
        ])
        dc_map = {
            **{net: f"{net}_dc" for net in bias_nets},
            **{pin: f"{net}_dc" for pin, net in main_pin_net_map.items()},
        }
        output_pins = [p for p in main_pins if p.lower().startswith("vout")]
        output_pin = output_pins[0] if output_pins else "Voutp"
        dc_map.update({
            "Vinn": "vfb_dc",
            "Vinp": "opin_dc",
            output_pin: "vout_dc",
            "vdd": "vdd_dc",
            "vss": "vss_dc",
        })
        lines.append(
            f'XDUT_DC ({" ".join(dc_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    dc_map,
                    main_pin_net_map,
                    suffix=f"_DC_{idx}",
                )
            )
        lines.extend([
            "CLOAD_DC (vout_dc 0) capacitor c=cload",
            "",
            "// PSRR+ path",
            "VDC_PSRR (opin_psrr 0) vsource dc=vcm type=dc",
        ])
        psrrp_map = path_net_map("PSRRP")
        lines.append(
            f'XDUT_PSRRP ({" ".join(psrrp_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    psrrp_map,
                    main_pin_net_map,
                    suffix=f"_PSRRP_{idx}",
                )
            )
        lines.extend([
            "CLOAD_PSRRP (vout_psrrp 0) capacitor c=cload",
            "",
            "// PSRR- path",
        ])
        psrrn_map = path_net_map("PSRRN")
        lines.append(
            f'XDUT_PSRRN ({" ".join(psrrn_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    psrrn_map,
                    main_pin_net_map,
                    suffix=f"_PSRRN_{idx}",
                )
            )
        lines.extend([
            "CLOAD_PSRRN (vout_psrrn 0) capacitor c=cload",
        ])
    else:
        output_pins = [p for p in main_pins if p.lower().startswith("vout")]
        out_n = output_pins[0] if len(output_pins) >= 1 else "Voutn"
        out_p = output_pins[1] if len(output_pins) >= 2 else "Voutp"
        lines.extend([
            "// ADM path",
            "VCM_ADM (vcm_adm 0) vsource dc=vcm type=dc",
            "VDIFF_ADM (vdiff_adm 0) vsource dc=vdiff_dc mag=vdiff_ac type=dc",
            "EINP_ADM (vinp_src vcm_adm vdiff_adm 0) vcvs gain=0.5",
            "EINN_ADM (vinn_src vcm_adm vdiff_adm 0) vcvs gain=-0.5",
            "LFBP_ADM (voutp_adm vinn_fb) inductor l=1T",
            "LFBN_ADM (voutn_adm vinp_fb) inductor l=1T",
            "CINP_ADM (vinp_fb vinp_src) capacitor c=1T",
            "CINN_ADM (vinn_fb vinn_src) capacitor c=1T",
        ])
        adm_map = path_net_map("ADM")
        lines.append(
            f'XDUT_ADM ({" ".join(adm_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    adm_map,
                    main_pin_net_map,
                    suffix=f"_ADM_{idx}",
                )
            )
        lines.extend([
            "CLOADP_ADM (voutp_adm 0) capacitor c=cload",
            "CLOADN_ADM (voutn_adm 0) capacitor c=cload",
            "",
            "// ACM path",
            "VCM_ACM (vcm_acm 0) vsource dc=vcm type=dc",
            "VDIFF_ACM (vdiff_acm 0) vsource dc=vdiff_dc mag=vdiff_ac type=dc",
            "EINP_ACM (vinp_src_cm vcm_acm vdiff_acm 0) vcvs gain=0.5",
            "EINN_ACM (vinn_src_cm vcm_acm vdiff_acm 0) vcvs gain=0.5",
            "LFBP_ACM (voutp_cm vinn_fb_cm) inductor l=1T",
            "LFBN_ACM (voutn_cm vinp_fb_cm) inductor l=1T",
            "CINP_ACM (vinp_fb_cm vinp_src_cm) capacitor c=1T",
            "CINN_ACM (vinn_fb_cm vinn_src_cm) capacitor c=1T",
        ])
        acm_map = path_net_map("ACM")
        lines.append(
            f'XDUT_ACM ({" ".join(acm_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    acm_map,
                    main_pin_net_map,
                    suffix=f"_ACM_{idx}",
                )
            )
        lines.extend([
            "CLOADP_ACM (voutp_cm 0) capacitor c=cload",
            "CLOADN_ACM (voutn_cm 0) capacitor c=cload",
            "",
            "// DC offset path (same topology as ADM, DC-only excitation)",
            "VCM_DC_OFFSET (vcm_dc 0) vsource dc=vcm type=dc",
            "VDIFF_DC_OFFSET (vdiff_dc_src 0) vsource dc=vdiff_dc type=dc",
            "EINP_DC (vinp_src_dc vcm_dc vdiff_dc_src 0) vcvs gain=0.5",
            "EINN_DC (vinn_src_dc vcm_dc vdiff_dc_src 0) vcvs gain=-0.5",
            "LFBP_DC (voutp_dc vinn_fb_dc) inductor l=1T",
            "LFBN_DC (voutn_dc vinp_fb_dc) inductor l=1T",
            "CINP_DC (vinp_fb_dc vinp_src_dc) capacitor c=1T",
            "CINN_DC (vinn_fb_dc vinn_src_dc) capacitor c=1T",
        ])
        dc_map = path_net_map("DC")   # own bias nets (LG_*_dc) and own supply pair, like the single-ended DC path
        dc_map.update({
            "Vinn": "vinn_fb_dc",
            "Vinp": "vinp_fb_dc",
            out_n: "voutn_dc",
            out_p: "voutp_dc",
            "vdd": "vdd_dc",
            "vss": "vss_dc",
        })
        lines.append(
            f'XDUT_DC ({" ".join(dc_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    dc_map,
                    main_pin_net_map,
                    suffix=f"_DC_{idx}",
                )
            )
        lines.extend([
            "CLOADP_DC (voutp_dc 0) capacitor c=cload",
            "CLOADN_DC (voutn_dc 0) capacitor c=cload",
            "",
            "// PSRR+ path",
        ])
        psrrp_map = path_net_map("PSRRP")
        lines.append(
            f'XDUT_PSRRP ({" ".join(psrrp_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    psrrp_map,
                    main_pin_net_map,
                    suffix=f"_PSRRP_{idx}",
                )
            )
        lines.extend([
            "CLOADP_PSRRP (voutp_psrrp 0) capacitor c=cload",
            "CLOADN_PSRRP (voutn_psrrp 0) capacitor c=cload",
            "",
            "// PSRR- path",
        ])
        psrrn_map = path_net_map("PSRRN")
        lines.append(
            f'XDUT_PSRRN ({" ".join(psrrn_map.get(pin, pin) for pin in main_pins)}) {main_subckt_name}'
        )
        for idx, inst in enumerate(support_instances, start=1):
            lines.append(
                rewrite_instance_line_with_pins(
                    inst,
                    signature_map.get(str(inst["subckt"]), []),
                    psrrn_map,
                    main_pin_net_map,
                    suffix=f"_PSRRN_{idx}",
                )
            )
        lines.extend([
            "CLOADP_PSRRN (voutp_psrrn 0) capacitor c=cload",
            "CLOADN_PSRRN (voutn_psrrn 0) capacitor c=cload",
        ])

    lines.extend([
        "",
        "saveOptions options save=allpub",
        "simulatorOptions options reltol=1e-3 vabstol=1e-6 iabstol=1e-12 temp=27 tnom=27",
        "ac ac start=1 stop=500G dec=100",   # from 1 Hz, see spectre_footer
        'dcOp dc write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status',
        "dcOpInfo info what=oppoint where=rawfile",
    ])
    return "\n".join(lines).strip()


def prepare_template_bundle(
    topo_name: str,
    content: str,
    kind_override: Optional[str] = None,
) -> Dict[str, object]:
    kind = kind_override or infer_template_kind(topo_name)
    global_params = extract_parameters_line(content)
    content_wo_params = re.sub(r'^\s*parameters\s+.*$', '', content, flags=re.MULTILINE)

    if kind == "simple":
        circuit_only, _ = split_circuit_and_testbench(content_wo_params)
        structure_body = circuit_only.strip() if circuit_only.strip() else content_wo_params.strip()
        subckt_params = extract_subckt_numeric(structure_body)
        structure_body = patch_core_with_param_names(structure_body, subckt_params)
        initial_params = {**global_params, **subckt_params}
        return {
            "kind": kind,
            "initial_params": initial_params,
            "structure_body": structure_body,
            "simulation_body": structure_body,
            "testbench": get_testbench_for_topology(topo_name),
            "main_subckt_name": topo_name,
        }

    signatures = parse_subckt_signatures(content_wo_params)
    top_instances = parse_top_level_instances(content_wo_params)
    signature = get_subckt_signature(content_wo_params, topo_name)
    if signature is None:
        for name, pins in signatures:
            if name.lower().startswith(topo_name.lower()):
                signature = (name, pins)
                break
    if signature is None:
        for inst in top_instances:
            if str(inst["name"]).upper() == "X0":
                maybe_main = get_subckt_signature(content_wo_params, str(inst["subckt"]))
                if maybe_main is not None:
                    signature = maybe_main
                    break
    if signature is None:
        for name, pins in signatures:
            if not name.lower().startswith("lg_"):
                signature = (name, pins)
                break
    if signature is None:
        raise ValueError(f"Could not find top-level subckt for complex OTA template '{topo_name}'")

    main_subckt_name, main_pins = signature
    signature_map = {name: pins for name, pins in signatures}
    all_blocks = [extract_subckt_block(content_wo_params, name) for name, _ in signatures]
    subckt_defs = "\n\n".join(block for block in all_blocks if block)
    main_instances = [inst for inst in top_instances if str(inst["subckt"]).lower() == main_subckt_name.lower()]
    support_instances = [inst for inst in top_instances if str(inst["subckt"]).lower() != main_subckt_name.lower()]
    if not main_instances:
        raise ValueError(f"Could not find top-level instance for complex OTA template '{topo_name}'")

    structure_body = (subckt_defs + "\n" + "\n".join(inst["raw"] for inst in top_instances)).strip()
    simulation_body = subckt_defs.strip()
    initial_params = dict(global_params)
    if "vdd" not in {k.lower(): v for k, v in initial_params.items()}:
        initial_params["vdd"] = "1.2"
    testbench = build_complex_ota_testbench(
        topo_name=topo_name,
        main_subckt_name=main_subckt_name,
        main_pins=main_pins,
        main_inst_nets=main_instances[0]["nets"],
        support_instances=support_instances,
        signature_map=signature_map,
        kind=kind,
    )
    return {
        "kind": kind,
        "initial_params": initial_params,
        "structure_body": structure_body,
        "simulation_body": simulation_body,
        "testbench": testbench,
        "main_subckt_name": main_subckt_name,
        "main_instance_name": str(main_instances[0]["name"]),
        "support_instance_names": [str(inst["name"]) for inst in support_instances],
    }

# --- PARAMETER EXTRACTION ---
def extract_parameters_line(text):
    m = re.search(r'^\s*parameters\s+(.+)$', text, re.MULTILINE)
    return dict(p.split('=') for p in m.group(1).split() if '=' in p) if m else {}

def extract_subckt_numeric(text):
    subckts = re.findall(r'subckt[\s\S]*?ends', text, re.IGNORECASE)
    param_map = {}
    for blk in subckts:
        for line in blk.splitlines():
            # 1. Width (w)
            wm = re.match(r'\s*(M\w+)[^\n]*\bw\s*=\s*([\d\.]+[fpnumkgt]*)', line, re.IGNORECASE)
            if wm: param_map[f"{wm.group(1)}_w"] = wm.group(2).lower()
            
            # 2. Length (l)
            lm = re.match(r'\s*(M\w+)[^\n]*\bl\s*=\s*([\d\.]+[fpnumkgt]*)', line, re.IGNORECASE)
            if lm: param_map[f"{lm.group(1)}_l"] = lm.group(2).lower()

            # 3. Resistor (r)
            rm = re.match(r'\s*(R\w+)[^\n]*\br\s*=\s*([\d\.]+[fpnumkgt]*)', line, re.IGNORECASE)
            if rm: param_map[f"{rm.group(1)}_r"] = rm.group(2).lower()
    return param_map

def split_circuit_and_testbench(content: str) -> tuple:
    """
    Split template content into circuit part and testbench.
    Circuit = all subckt...ends blocks + top-level subcircuit instances (X0, X1, ...) that
    connect the main and bias blocks. Testbench = sources, loads, and measurement (R0, V*, C*, etc.).

    Returns:
        (circuit_only, testbench) - circuit_only may be empty if no subckts; testbench may be empty.
    """
    lines = content.strip().splitlines()
    circuit_lines = []
    top_level_other = []  # lines after subckts: X*, R*, V*, C*, etc.
    subckt_names = set()
    in_subckt = False

    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            if in_subckt:
                circuit_lines.append(line)
            else:
                top_level_other.append(line)
            continue
        if s.lower().startswith("subckt"):
            in_subckt = True
            circuit_lines.append(line)
            # subckt name is second token
            parts = s.split()
            if len(parts) >= 2:
                subckt_names.add(parts[1])
        elif s.lower().startswith("ends"):
            in_subckt = False
            circuit_lines.append(line)
        elif in_subckt:
            circuit_lines.append(line)
        else:
            top_level_other.append(line)

    # Top-level lines: subcircuit instances (X... <subckt_name>) go to circuit; rest to testbench
    circuit_other_lines = []
    testbench_lines = []
    for line in top_level_other:
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("#"):
            testbench_lines.append(line)
            continue
        # Check if this line is a subcircuit call: X<id> ( nodes ) <subckt_type>
        if s.upper().startswith("X"):
            # After ')', first token is subckt type (may be followed by key=value)
            match = re.search(r'\)\s+(\w+)', s)
            if match:
                inst_type = match.group(1)
                if inst_type in subckt_names:
                    circuit_other_lines.append(line)
                    continue
        testbench_lines.append(line)

    circuit_only = "\n".join(circuit_lines + circuit_other_lines).strip() if (circuit_lines or circuit_other_lines) else ""
    testbench = "\n".join(testbench_lines).strip() if testbench_lines else ""
    return circuit_only, testbench


def patch_core_with_param_names(text, param_map):
    for pname, val in param_map.items():
        inst = pname[:-2]
        # Regex pattern depends on suffix (_w, _l, _r)
        if pname.endswith('_w'):
            pattern = rf'(\b{inst}\b[^\n]*?\bw\s*=\s*){re.escape(val)}(?=\s|$)'
        elif pname.endswith('_l'):
            pattern = rf'(\b{inst}\b[^\n]*?\bl\s*=\s*){re.escape(val)}(?=\s|$)'
        elif pname.endswith('_r'):
            pattern = rf'(\b{inst}\b[^\n]*?\br\s*=\s*){re.escape(val)}(?=\s|$)'
        else:
            continue
            
        text = re.sub(pattern, rf'\1{pname}', text, flags=re.IGNORECASE)
    return text

_SI_EXACT_UNITS = ((9, 'g'), (6, 'meg'), (3, 'k'), (0, ''),
                   (-3, 'm'), (-6, 'u'), (-9, 'n'), (-12, 'p'), (-15, 'f'))


def format_si_exact(v_str: str) -> str:
    """Losslessly re-render a numeric literal in SI-suffix notation via Decimal.

    "1.3e-06" -> "1.3u", "2.41e-07" -> "241n", "8.2k" -> "8.2k" (unlike format_si, no quantization). Returns the input
    unchanged when it is not a plain number / SI token or falls outside the f..g suffix range.
    """
    from decimal import Decimal, InvalidOperation
    t = str(v_str).strip()
    m = re.fullmatch(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(meg|[fpnumkg]?)', t)
    if not m:
        return v_str
    num, suffix = m.groups()
    suffix_scale = {'f': -15, 'p': -12, 'n': -9, 'u': -6, 'm': -3,
                    '': 0, 'k': 3, 'meg': 6, 'g': 9}[suffix.lower()]
    try:
        d = Decimal(num).scaleb(suffix_scale)
    except InvalidOperation:
        return v_str
    if d == 0:
        return "0"
    exp10 = d.adjusted()          # exponent of the most significant digit
    for k, unit in _SI_EXACT_UNITS:
        if exp10 >= k:            # mantissa lands in [1, 1000)
            mant = d.scaleb(-k)
            s = format(mant.normalize(), 'f')
            return s + unit
    return v_str                  # below 1f — keep original


def wrap_netlist(core_body, sample_params, analysis_kind: Optional[str] = None):
    def _fmt(v):
        if isinstance(v, str):
            # normalize e-notation strings (incl. hybrids like '2.706e+09u', which Spectre rejects) to SI-suffix
            # notation.
            return format_si_exact(v)
        return format_si(v)
    param_line = ' '.join(f"{k}={_fmt(v)}" for k, v in sample_params.items())
    header = spectre_header.format(
        parameters_line=param_line,
        tech_model=resolve_tech_model(),
    )
    body = core_body.strip()
    lower_body = body.lower()
    has_embedded_analyses = (
        "ac ac " in lower_body or
        "\nac ac " in lower_body or
        "dcop dc" in lower_body
    )
    analysis_header = f"// CIRCUITOP_ANALYSIS_KIND: {analysis_kind}\n" if analysis_kind else ""
    if has_embedded_analyses:
        return header + "\n" + analysis_header + body + "\n"
    return header + "\n" + analysis_header + body + "\n" + spectre_footer

