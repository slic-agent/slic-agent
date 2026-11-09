"""Readable names of the released topologies for the browser UI: topology id -> (circuit family, input-device type
NMOS / PMOS). The page lists families with the input type as a switch; the id stays the key everywhere else. A topology
not in the table (e.g. your own circuits from `slic tasks`) is shown under its id with no input switch."""
from __future__ import annotations

from typing import Dict, Tuple

FAMILIES: Dict[str, Tuple[str, str]] = {
    # simple amplifiers
    "common_source_amplifier_with_resistive_load": ("Common-source amplifier, resistive load", "NMOS"),
    "common_source_amplifier_with_resistive_load_pmos_input": ("Common-source amplifier, resistive load", "PMOS"),
    "common_source_amplifier_with_current_source_load": ("Common-source amplifier, current-source load", "NMOS"),
    "common_source_amplifier_with_current_source_load_pmos_input": ("Common-source amplifier, current-source load", "PMOS"),
    "common_source_amplifier_with_pmos_diode_connected_load": ("Common-source amplifier, complementary diode load", "NMOS"),
    "common_source_amplifier_with_diode_connected_nmos_load_pmos_input": ("Common-source amplifier, complementary diode load", "PMOS"),
    "common_source_amplifier_with_diode_connected_nmos_load": ("Common-source amplifier, same-type diode load", "NMOS"),
    "common_source_resistive_degeneration_with_resistive_load": ("Common-source amplifier, source degeneration, resistive load", "NMOS"),
    "common_source_resistive_degeneration_with_resistive_load_pmos_input": ("Common-source amplifier, source degeneration, resistive load", "PMOS"),
    "common_source_resistive_source_degeneration_with_current_source_load": ("Common-source amplifier, source degeneration, current-source load", "NMOS"),
    "common_source_resistive_source_degeneration_with_current_source_load_pmos_input": ("Common-source amplifier, source degeneration, current-source load", "PMOS"),
    "common_source_with_pmos_cascode_current_source_load": ("Common-source amplifier, cascode current-source load", "NMOS"),
    "common_source_with_nmos_cascode_current_source_load_pmos_input": ("Common-source amplifier, cascode current-source load", "PMOS"),
    "cascode_stage_with_current_source_load": ("Cascode amplifier, current-source load", "NMOS"),
    "cascode_stage_with_current_source_load_pmos_input": ("Cascode amplifier, current-source load", "PMOS"),
    "common_gate_amplifier_with_resistive_load": ("Common-gate amplifier, resistive load", "NMOS"),
    "common_gate_amplifier_with_resistive_load_pmos_input": ("Common-gate amplifier, resistive load", "PMOS"),
    "common_gate_amplifier_with_current_source_load": ("Common-gate amplifier, current-source load", "NMOS"),
    # single-ended OTAs
    "single_ended_LG_nmos": ("Five-transistor OTA", "NMOS"),
    "single_ended_pmos_LG_pmos": ("Five-transistor OTA", "PMOS"),
    "single_ended_current_mirror_LG_nmos": ("Current-mirror OTA", "NMOS"),
    "single_ended_current_mirror_pmos_LG_pmos": ("Current-mirror OTA", "PMOS"),
    "single_ended_cascode_current_mirror_LG_nmos_l1_biasn_LV": ("Cascode current-mirror OTA", "NMOS"),
    "single_ended_cascode_current_mirror_pmos_LG_pmos_l1_biasp_LV": ("Cascode current-mirror OTA", "PMOS"),
    "single_ended_cascode_LG_nmos_l1_biasn_biasp_LV": ("Folded-cascode OTA", "NMOS"),
    "single_ended_cascode_pmos_LG_pmos_l1_biasn_biasp_LV": ("Folded-cascode OTA", "PMOS"),
    "single_ended_telescopic_LG_nmos_biasp_LV": ("Telescopic OTA, cascoded mirror load", "NMOS"),
    "single_ended_telescopic_pmos_LG_pmos_biasn_LV": ("Telescopic OTA, cascoded mirror load", "PMOS"),
    "single_ended_telescopic_cascode_LG_nmos_l1_biasn_biasp_LV": ("Telescopic cascode OTA", "NMOS"),
    "single_ended_telescopic_cascode_pmos_LG_nmos_l1_biasn_biasp_LV": ("Telescopic cascode OTA", "PMOS"),
    "single_ended_miller_compensated_LG_nmos": ("Two-stage Miller-compensated OTA", "NMOS"),
    "single_ended_miller_compensated_pmos_LG_pmos": ("Two-stage Miller-compensated OTA", "PMOS"),
    "single_ended_two_stage_LG_nmos": ("Two-stage OTA, mirror-load first stage", "NMOS"),
    "single_ended_two_stage_LG_pmos_l1": ("Two-stage OTA, current-source-load first stage", "NMOS"),
    # fully differential OTAs
    "fully_differential_LG_pmos_l1": ("Fully differential five-transistor OTA", "NMOS"),
    "fully_differential_pmos_LG_nmos_l1": ("Fully differential five-transistor OTA", "PMOS"),
    "fully_differential_cascode_LG_load_biasp_LV_nmos_l1": ("Fully differential cascode OTA", "NMOS"),
    "fully_differential_cascode_pmos_LG_load_biasn_LV_pmos_l1": ("Fully differential cascode OTA", "PMOS"),
    "Fully_differential_folded_cascode_LG_nmos_l1_biasn_biasp_LV": ("Fully differential folded-cascode OTA", "NMOS"),
    "Fully_differential_folded_cascode_pmos_LG_pmos_l1_biasn_biasp_LV": ("Fully differential folded-cascode OTA", "PMOS"),
    "Fully_differential_current_mirror_LG_nmos": ("Fully differential current-mirror OTA", "NMOS"),
}


def describe(topology: str) -> Dict[str, str]:
    """{"id", "family", "input"} of a topology id; an unknown id is its own family with input "" (no switch)."""
    family, inp = FAMILIES.get(topology, (topology, ""))
    return {"id": topology, "family": family, "input": inp}
