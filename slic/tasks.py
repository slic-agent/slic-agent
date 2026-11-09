"""`slic tasks --spec tasks.yaml` — build a task parquet in the released row format from your own circuits.

Use it to continue RL (or evaluate) on your own circuits or technology card. Rows are built with the environment's own code
(graph_serializer for the topology XML and action space, state_utils.construct_netlist_from_dataset for the netlist, the
configured simulator for the starting metrics); the output has the released columns `data_source` and `extra_info` (docs/DATASETS.md).

Spec file (YAML or JSON):

  tech_node: 65nm                 # required: geometry box and card (22nm 32nm 45nm 65nm 90nm 130nm), CIRCUITOP_TECH_NODE
  # tech_model: pdk/my_card.scs   # optional: your own Spectre-syntax card (CIRCUITOP_TECH_MODEL); tech_node still sets the
  #                               # geometry box. Relative paths are relative to the spec file.
  output: data/my_tasks/train.parquet   # or --out
  split: train                    # optional default for tasks that do not set one (default: train)
  goal_level: custom              # optional default label (default: custom)
  tasks:
    - topology: common_source_amplifier_with_current_source_load     # the name selects the test bench (see below)
      netlist: |                  # sub-circuits + top-level X instances, device sizes as parameter names
        subckt cs_amp vb vdd vin vout vss
            M0 (vout vin vss vss) nch w=M0_w l=M0_l
            M1 (vout vb vdd vdd) pch w=M1_w l=M1_l
        ends cs_amp
        ...
        X0 (vb vdd vb_in vout 0) cs_amp
        X1 (vb vdd 0) cs_amp_bias
      # netlist_file: circuits/cs_amp.scs   # instead of netlist (relative to the spec file)
      params: {M0_w: 1.2u, M0_l: 180n, M1_w: 2u, M1_l: 200n, iref: 5u, vdd: 1.2}   # every referenced parameter + vdd
      target: {gain_min: 30, power_max: 100, bandwidth_min: 1.0, phase_margin_min_deg: 30, phase_margin_max_deg: 90}
      task_id: cs_amp_0           # optional (default <topology>__<index:05d>); must be unique
      split: train                # optional
      goal_level: easy            # optional label
      index: 0                    # optional (default: position in the list)
      metadata: {circuit_kind: simple}   # optional; inferred from the topology name otherwise

Circuit contract (the environment appends one of three fixed test benches, chosen like this):
  * simple (single-stage) circuits: the top level connects the input to `vb_in` (`vin` for a topology whose name starts
    with `common_gate_amplifier_`), the output to `vout`, the supply to `vdd` and ground to `0`.
  * complex single-ended / fully-differential OTAs: the topology name contains `single_ended` or `fully_differential`
    (or metadata.circuit_kind says so); the main sub-circuit has pins Vinn, Vinp, Vout..., vdd, vss and is instantiated as
    X0; bias generators are further X instances whose bias nets are named LG_*.
  * MOSFETs use the models `nch` / `pch`; device sizes (w, l, r, c, dc) are parameter names, not numbers; no `parameters`,
    `include`, `simulator` or analysis lines (the environment writes them).
  Use a released row of a similar circuit as the reference (`slic data --family standard`).

Each starting design is simulated once, like the environment's reset, and labelled `invalid` when the simulation fails
(any error other than a design error), the gain is <= 0 dB or not finite, or the in-band flatness is missing or > 6 dB;
otherwise `valid`. A design outside the node's geometry box is rejected (the environment would start it in a design error).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._common import REPO_ROOT, ConfigError

DATA_SOURCE = "circuit_design"
TARGET_KEYS = ("gain_min", "power_max", "bandwidth_min", "phase_margin_min_deg", "phase_margin_max_deg")
METRIC_KEYS = ("gain_dB", "power_uW", "bandwidth_MHz", "bandwidth_kind", "inband_flatness_db", "phase_margin_deg",
               "phase_margin_status")
KINDS = ("simple", "complex_single_ended", "complex_fully_differential")
PERCENTAGE_LIMITS = {"percentage_min": 1, "percentage_max": 100}   # step limits of every released task
SPEC_KEYS = {"tech_node", "tech_model", "output", "split", "goal_level", "tasks"}
TASK_KEYS = {"topology", "netlist", "netlist_file", "params", "target", "task_id", "split", "goal_level", "index", "metadata"}
SIZE_KEYS = ("w", "l", "r", "c", "dc")
FORBIDDEN_LINE = re.compile(r"^\s*(parameters|include|simulator|global|ahdl_include|dcop|dcopinfo|ac\s|tran\s|"
                            r"simulatoroptions|saveoptions)\b", re.I)
IDENT = re.compile(r"(?<![\w.])[A-Za-z_]\w*")   # a name, not the unit suffix / exponent of a number (1.2u, 1e-6)
KV = re.compile(r"(\w+)\s*=\s*([^\s]+)")
SIM_ATTEMPTS = 3   # the environment retries its reset-time simulation up to three times when it returns no operating point
ENV_PKG = REPO_ROOT / "agent_system" / "environments" / "env_package" / "circuitop"
DATA_PRE = REPO_ROOT / "circuit" / "data_pre"


_NOTES: Dict[str, int] = {}


def _note(msg: str) -> None:
    if msg:
        _NOTES[msg] = _NOTES.get(msg, 0) + 1


class TaskError(ConfigError):
    def __init__(self, where: str, msg: str):
        super().__init__(f"{where}: {msg}")


# ----------------------------------------------------------------------------------------------------------------------
# environment setup and the environment's own modules
# ----------------------------------------------------------------------------------------------------------------------
def _setup_env(tech_node: str, tech_model: Optional[str]) -> None:
    """Simulator / technology variables, exactly as the environment reads them (set before the modules are imported)."""
    os.environ["CIRCUITOP_TECH_NODE"] = tech_node
    if tech_model:
        os.environ["CIRCUITOP_TECH_MODEL"] = tech_model
    else:
        os.environ.pop("CIRCUITOP_TECH_MODEL", None)   # the spec is authoritative: no card override from the shell
    os.environ.setdefault("CIRCUIT_SIM_BACKEND", "ngspice")
    os.environ.setdefault("NGSPICE_BIN", "ngspice")
    os.environ.setdefault("NGSPICE_AC_FOLLOW_SPECTRE", "1")   # the training / evaluation launchers' default


def _modules():
    for p in (str(DATA_PRE), str(ENV_PKG)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import netlist_gen
    from graph_serializer import GraphXMLGenerator
    from circuitop import state_utils
    return netlist_gen, GraphXMLGenerator, state_utils


def _make_runner(scratch: str):
    """The simulator class the environment instantiates for CIRCUIT_SIM_BACKEND (circuitop/env.py)."""
    _modules()
    backend = os.environ.get("CIRCUIT_SIM_BACKEND", "spectre").strip().lower()
    if backend == "ngspice":
        from ngspice_runner import NgspiceRunner as cls
    elif backend == "spectre":
        from spectre_runner import SpectreRunner as cls
    else:
        raise ConfigError(f"unknown CIRCUIT_SIM_BACKEND={backend!r} (ngspice|spectre)")
    os.makedirs(scratch, exist_ok=True)
    return cls(scratch_dir=scratch)


def _preflight(state_utils) -> str:
    card = state_utils.resolve_tech_model()
    if not os.path.isfile(card):
        hint = " (the 22-130 nm cards come with `slic data --cards`)" if not os.environ.get("CIRCUITOP_TECH_MODEL") else ""
        raise ConfigError(f"technology card not found: {card}{hint}")
    backend = os.environ["CIRCUIT_SIM_BACKEND"].strip().lower()
    if backend == "ngspice":
        ng = os.environ["NGSPICE_BIN"]
        if not (shutil.which(ng) or (os.path.isfile(ng) and os.access(ng, os.X_OK))):
            raise ConfigError(f"ngspice binary not found: NGSPICE_BIN={ng} (docs/SIMULATORS.md)")
    return card


# ----------------------------------------------------------------------------------------------------------------------
# spec parsing and validation (no simulation)
# ----------------------------------------------------------------------------------------------------------------------
def _load_spec(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.is_file():
        raise ConfigError(f"spec file not found: {path}")
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)   # JSON is valid YAML
    if not isinstance(spec, dict):
        raise ConfigError(f"{path}: the spec must be a mapping with a `tasks:` list")
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {sorted(unknown)}; allowed: {sorted(SPEC_KEYS)}")
    if not isinstance(spec.get("tasks"), list) or not spec["tasks"]:
        raise ConfigError(f"{path}: `tasks:` must be a non-empty list")
    return spec


def _param_value(where: str, name: str, v: Any, netlist_gen) -> str:
    if isinstance(v, bool) or v is None:
        raise TaskError(where, f"params.{name} must be a number or an SI string such as 1.2u, got {v!r}")
    if isinstance(v, (int, float)):
        if not math.isfinite(float(v)):
            raise TaskError(where, f"params.{name} is not finite: {v!r}")
        # 12 significant digits (drops binary noise such as 6.799999999999999e-06), then an exact SI rendering
        # (netlist_gen.format_si would quantise the mantissa to 0.1)
        s = netlist_gen.format_si_exact(format(float(v), ".12g"))
    elif isinstance(v, str):
        s = v.strip()
    else:
        raise TaskError(where, f"params.{name} must be a number or an SI string, got {type(v).__name__}")
    try:
        netlist_gen.parse_si(s)
    except ValueError:
        raise TaskError(where, f"params.{name}={v!r} is not a number / SI value (e.g. 1.2u, 180n, 3.3k, 1e-6)")
    return s


def _meaningful_lines(text: str) -> List[str]:
    return sorted(ln.strip() for ln in text.splitlines()
                  if ln.strip() and not ln.strip().startswith(("//", "#", "*")))


def _referenced_params(netlist: str) -> List[str]:
    """Parameter names used as device sizes (w / l / r / c / dc) inside the netlist."""
    names = set()
    for ln in netlist.splitlines():
        s = ln.strip()
        if not s or s.startswith(("//", "#", "*")) or s.lower().startswith(("subckt", "ends")):
            continue
        tail = s.split(")", 1)[-1] if ")" in s else s
        for k, v in KV.findall(tail):
            if k.lower() in SIZE_KEYS:
                names.update(IDENT.findall(v))
    return sorted(names)


def _numeric_sizes(netlist: str) -> List[str]:
    out = []
    for ln in netlist.splitlines():
        s = ln.strip()
        if not s or s.startswith(("//", "#", "*")) or s.lower().startswith(("subckt", "ends")):
            continue
        tail = s.split(")", 1)[-1] if ")" in s else s
        for k, v in KV.findall(tail):
            if k.lower() in SIZE_KEYS and not IDENT.search(v):
                out.append(f"{s.split()[0]}: {k}={v}")
    return out


def _check_testbench_contract(where: str, topology: str, kind: str, main: str, structure: str, netlist_gen) -> None:
    """The nets / pins the environment's test bench for this kind connects to (circuit/data_pre/netlist_gen.py)."""
    if kind == "simple":
        top_nets = {n for inst in netlist_gen.parse_top_level_instances(structure) for n in inst["nets"]}
        inp = "vin" if topology.lower().startswith("common_gate_amplifier_") else "vb_in"
        missing = [n for n in (inp, "vout", "vdd") if n not in top_nets]
        if missing:
            raise TaskError(where, f"a simple circuit's top-level X instances must connect the nets {missing} (the test bench "
                                   f"drives `{inp}`, loads `vout`, supplies `vdd`; ground is `0`)")
        return
    sig = netlist_gen.get_subckt_signature(structure, main)
    pins = list(sig[1]) if sig else []
    outs = [p for p in pins if p.lower().startswith("vout")]
    need_outs = 2 if kind == "complex_fully_differential" else 1
    missing = [p for p in ("Vinn", "Vinp", "vdd", "vss") if p not in pins]
    if missing or len(outs) < need_outs:
        raise TaskError(where, f"the main sub-circuit {main} of a {kind} circuit needs the pins Vinn, Vinp, vdd, vss and "
                               f"{need_outs} output pin(s) named Vout...; it has {pins}")


def _metadata(where: str, topology: str, netlist: str, given: Dict[str, Any], netlist_gen) -> Tuple[Dict[str, str], Dict]:
    """complexity / signal_mode / circuit_kind, inferred the way the released task builders did."""
    if not isinstance(given, dict):
        raise TaskError(where, "metadata must be a mapping")
    unknown = set(given) - {"complexity", "signal_mode", "circuit_kind"}
    if unknown:
        raise TaskError(where, f"unknown metadata key(s) {sorted(unknown)}")
    kind = given.get("circuit_kind") or netlist_gen.infer_template_kind(topology)
    if kind not in KINDS:
        raise TaskError(where, f"metadata.circuit_kind={kind!r}; choices: {list(KINDS)}")
    try:
        bundle = netlist_gen.prepare_template_bundle(topology, netlist, kind_override=kind)
    except Exception as exc:   # the environment would fail the same way at reset
        raise TaskError(where, f"the netlist does not fit the {kind} test bench: {type(exc).__name__}: {exc}")
    main = str(bundle["main_subckt_name"])
    _check_testbench_contract(where, topology, kind, main, str(bundle["structure_body"]), netlist_gen)
    md = {
        "complexity": "complex" if kind.startswith("complex_") else "simple",
        "signal_mode": ("fully_differential" if kind == "complex_fully_differential" or "fully_differential" in main.lower()
                        else "single_ended"),
        "circuit_kind": kind,
    }
    for k in ("complexity", "signal_mode"):
        if k in given and given[k] != md[k]:
            raise TaskError(where, f"metadata.{k}={given[k]!r} contradicts circuit_kind={kind!r} (expected {md[k]!r})")
    return md, bundle


def _build_task(i: int, t: Dict[str, Any], spec: Dict[str, Any], base: Path, modules) -> Dict[str, Any]:
    netlist_gen, GraphXMLGenerator, state_utils = modules
    where = f"tasks[{i}]"
    if not isinstance(t, dict):
        raise TaskError(where, "each task must be a mapping")
    unknown = set(t) - TASK_KEYS
    if unknown:
        raise TaskError(where, f"unknown key(s) {sorted(unknown)}; allowed: {sorted(TASK_KEYS)}")
    topology = t.get("topology")
    if not isinstance(topology, str) or not re.fullmatch(r"[A-Za-z_]\w*", topology):
        raise TaskError(where, f"topology must be an identifier-like name, got {topology!r}")
    where = f"tasks[{i}] ({topology})"

    # -- netlist -------------------------------------------------------------------------------------------------------
    if ("netlist" in t) == ("netlist_file" in t):
        raise TaskError(where, "give exactly one of `netlist` (text) and `netlist_file` (path)")
    if "netlist_file" in t:
        p = Path(os.path.expanduser(str(t["netlist_file"])))
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            raise TaskError(where, f"netlist_file not found: {p}")
        netlist = p.read_text(encoding="utf-8")
    else:
        netlist = t["netlist"]
    if not isinstance(netlist, str) or "subckt" not in netlist.lower():
        raise TaskError(where, "the netlist must contain the circuit's `subckt ... ends` blocks")
    bad = [ln.strip() for ln in netlist.splitlines() if FORBIDDEN_LINE.match(ln)]
    if bad:
        raise TaskError(where, f"the netlist must hold only the circuit (the environment writes parameters, model include, "
                               f"test bench and analyses); remove: {bad[:3]}")
    numeric = _numeric_sizes(netlist)
    if numeric:
        raise TaskError(where, f"device sizes must be parameter names (so the agent can change them), found numbers: "
                               f"{numeric[:5]}")
    md, bundle = _metadata(where, topology, netlist, t.get("metadata") or {}, netlist_gen)
    structure_netlist = str(bundle["structure_body"])
    if _meaningful_lines(structure_netlist) != _meaningful_lines(netlist):
        dropped = sorted(set(_meaningful_lines(netlist)) - set(_meaningful_lines(structure_netlist)))
        raise TaskError(where, f"the netlist holds lines outside the circuit (only sub-circuits and the top-level X instances "
                               f"of them are kept): {dropped[:5]}")

    # -- params --------------------------------------------------------------------------------------------------------
    params_in = t.get("params")
    if not isinstance(params_in, dict) or not params_in:
        raise TaskError(where, "params must be a non-empty mapping {parameter: value}")
    params = {str(k): _param_value(where, str(k), v, netlist_gen) for k, v in params_in.items()}
    needed = set(_referenced_params(structure_netlist)) | {"vdd"}
    missing, extra = sorted(needed - set(params)), sorted(set(params) - needed)
    if missing:
        raise TaskError(where, f"params lacks {missing} (every parameter the netlist uses, plus vdd, needs a starting value)")
    if extra:   # harmless (they only reach the parameters line), but usually a naming slip
        _note(f"{where}: params {extra} are not used by the netlist (kept as given)")

    # -- target --------------------------------------------------------------------------------------------------------
    target_in = t.get("target")
    if not isinstance(target_in, dict):
        raise TaskError(where, f"target must be a mapping with {list(TARGET_KEYS)}")
    if set(target_in) != set(TARGET_KEYS):
        raise TaskError(where, f"target needs exactly {list(TARGET_KEYS)}; missing {sorted(set(TARGET_KEYS) - set(target_in))}"
                               f", unknown {sorted(set(target_in) - set(TARGET_KEYS))}")
    target = {}
    for k in TARGET_KEYS:
        v = target_in[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise TaskError(where, f"target.{k} must be a finite number, got {v!r}")
        target[k] = float(v)
    if target["phase_margin_min_deg"] > target["phase_margin_max_deg"]:
        raise TaskError(where, "target.phase_margin_min_deg > phase_margin_max_deg")
    if target["power_max"] <= 0 or target["bandwidth_min"] < 0:
        raise TaskError(where, "target.power_max must be > 0 (uW) and bandwidth_min >= 0 (MHz)")

    # -- labels --------------------------------------------------------------------------------------------------------
    index = t.get("index", i)
    if isinstance(index, bool) or not isinstance(index, int):
        raise TaskError(where, f"index must be an integer, got {index!r}")
    labels = {}
    for k, default in (("split", spec.get("split", "train")), ("goal_level", spec.get("goal_level", "custom")),
                       ("task_id", f"{topology}__{index:05d}")):
        v = t.get(k, default)
        if not isinstance(v, str) or not v.strip():
            raise TaskError(where, f"{k} must be a non-empty string, got {v!r}")
        labels[k] = v

    # -- topology XML and action space: the serializer the released task builders used -------------------------------
    serializer = GraphXMLGenerator()
    serial = "parameters " + " ".join(f"{k}={v}" for k, v in params.items()) + "\n" + structure_netlist
    xml, handle2id, nodes = serializer.generate_static_structure(serial, str(bundle["main_subckt_name"]))
    id2handle = {rid: h for h, rid in handle2id.items()}
    spaces = serializer.generate_admissible_actions(nodes, id2handle, serial)   # xml (handles) and netlist (device ids)
    actions, net_actions = spaces["xml"], spaces["netlist"]
    if not actions["valid_targets"]:
        raise TaskError(where, "the circuit has no tunable device (MOSFET, current source, sub-circuit R / C)")
    actions["constraints"] = dict(PERCENTAGE_LIMITS)
    net_actions["constraints"] = dict(PERCENTAGE_LIMITS)

    task_data = {
        "structure": {"netlist": structure_netlist, "xml": xml},
        "admissible_actions": {"xml": actions, "netlist": net_actions},
        "initial_state": {"metrics": None, "params": params},
        "metadata": md,
    }
    # the netlist the environment simulates at reset, and its geometry pre-check
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # anything the netlist builder prints becomes a note of this task
        sim_netlist = state_utils.construct_netlist_from_dataset(
            {**task_data, "metadata": {**md, "topology_name": topology}}, representation="xml")
    for ln in buf.getvalue().splitlines():
        _note(ln.strip())
    violations = state_utils.find_geometry_bound_violations(sim_netlist, topology=topology)
    if violations:
        raise TaskError(where, f"starting design outside the {os.environ['CIRCUITOP_TECH_NODE']} geometry box "
                               f"({state_utils.format_tech_geometry_bounds(topology=topology)}): {violations[:5]}")
    return {
        "row": {
            "data_source": DATA_SOURCE,
            "extra_info": {
                "task_id": labels["task_id"], "split": labels["split"], "index": index, "goal_level": labels["goal_level"],
                "initial_state_class": None,
                "env_kwargs": {"topology": topology, "target": target, "task_data": task_data},
            },
        },
        "sim_netlist": sim_netlist,
    }


# ----------------------------------------------------------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------------------------------------------------------
def classify(gain: Any, inband: Any, error_type: Any) -> str:
    """valid / invalid starting state (the rule of every released task set)."""
    if error_type not in (None, "", "design_error"):
        return "invalid"
    try:
        g = float(gain)
    except (TypeError, ValueError):
        return "invalid"
    if not math.isfinite(g) or g <= 0:
        return "invalid"
    try:
        ib = float(inband)
    except (TypeError, ValueError):
        return "invalid"
    return "valid" if math.isfinite(ib) and ib <= 6.0 else "invalid"


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


_RUNNER = None


def _worker_init(env: Dict[str, str], scratch_root: str) -> None:
    global _RUNNER
    os.environ.update(env)
    _RUNNER = _make_runner(tempfile.mkdtemp(prefix="w_", dir=scratch_root))


def _simulate(sim_netlist: str) -> Dict[str, Any]:
    res = {}
    for _ in range(SIM_ATTEMPTS):
        res = _RUNNER.run_simulation(sim_netlist, timeout=900, threads=1)
        if res.get("op_data"):
            break
    if res.get("error_type") in ("infrastructure_error", "python_exception"):
        raise RuntimeError(f"simulator infrastructure failure: {res.get('error_type')}: {res.get('feedback')}")
    power = res.get("power")
    metrics = {
        "gain_dB": _plain(res.get("gain")),
        "power_uW": float(power) * 1e6 if power is not None else None,
        "bandwidth_MHz": _plain(res.get("bandwidth_MHz")),
        "bandwidth_kind": res.get("bandwidth_kind"),
        "inband_flatness_db": _plain(res.get("inband_flatness_db")),
        "phase_margin_deg": _plain(res.get("phase_margin_deg")),
        "phase_margin_status": res.get("phase_margin_status"),
    }
    return {"metrics": metrics, "error_type": res.get("error_type"), "feedback": res.get("feedback") or "",
            "class": classify(metrics["gain_dB"], metrics["inband_flatness_db"], res.get("error_type"))}


# ----------------------------------------------------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------------------------------------------------
def _plain_tree(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain_tree(v) for k, v in x.items()}
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        return _plain_tree(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_plain_tree(v) for v in x]
    return x


def _prune(x: Any) -> Any:
    """drop the None fields parquet adds for keys that exist only in other rows (struct union)"""
    if isinstance(x, dict):
        return {k: _prune(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_prune(v) for v in x]
    return x


def _write(rows: List[Dict[str, Any]], out: Path) -> None:
    import json

    import pandas as pd

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    pd.DataFrame({"data_source": [r["data_source"] for r in rows],
                  "extra_info": [r["extra_info"] for r in rows]}).to_parquet(tmp, index=False)
    back = pd.read_parquet(tmp)
    key = lambda x: json.dumps(_prune(_plain_tree(x)), sort_keys=True, default=str)
    if list(back.columns) != ["data_source", "extra_info"] or len(back) != len(rows) or any(
            key(dict(b)) != key(r["extra_info"]) for b, r in zip(back["extra_info"], rows)):
        tmp.unlink()
        raise RuntimeError(f"{out}: the parquet round trip changed a row; nothing written")
    os.replace(tmp, out)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic tasks", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="task spec (YAML or JSON), format above")
    ap.add_argument("--out", default=None, help="output parquet (overrides `output:` in the spec)")
    ap.add_argument("--workers", type=int, default=1, help="parallel simulations (default 1)")
    ap.add_argument("--scratch", default=None, help="simulator scratch directory (default: a temporary directory)")
    ap.add_argument("--check", action="store_true", help="validate the spec and build structures only; no simulation, no output")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("--workers must be >= 1")

    spec_path = Path(a.spec).expanduser().resolve()
    spec = _load_spec(spec_path)
    base = spec_path.parent
    tech_node = spec.get("tech_node")
    if not isinstance(tech_node, str) or not tech_node:
        raise ConfigError(f"{spec_path}: `tech_node:` is required (it selects the geometry box and, without tech_model, the card)")
    tech_model = spec.get("tech_model")
    if tech_model is not None:
        p = Path(os.path.expanduser(str(tech_model)))
        tech_model = str(p if p.is_absolute() else (base / p).resolve())
    _setup_env(tech_node, tech_model)
    modules = _modules()
    if tech_node not in modules[2].TECH_MODEL_MAP:
        raise ConfigError(f"tech_node={tech_node!r}; choices: {sorted(modules[2].TECH_MODEL_MAP)} "
                          "(for your own card, pick the closest node for the geometry box and set tech_model)")
    out = None
    if not a.check:
        out_s = a.out or spec.get("output")
        if not out_s:
            raise ConfigError("no output path: set `output:` in the spec or pass --out")
        out = Path(os.path.expanduser(str(out_s)))
        out = out if out.is_absolute() else Path.cwd() / out
        if out.suffix != ".parquet":
            raise ConfigError(f"output must be a .parquet file, got {out}")
        if out.exists() and not a.overwrite:
            raise ConfigError(f"{out} exists; pass --overwrite to replace it")

    built = [_build_task(i, t, spec, base, modules) for i, t in enumerate(spec["tasks"])]
    ids = Counter(b["row"]["extra_info"]["task_id"] for b in built)
    dup = sorted(k for k, n in ids.items() if n > 1)
    if dup:
        raise ConfigError(f"duplicate task_id(s): {dup[:5]}")
    for msg, n in _NOTES.items():
        print(f"[slic] note: {msg}" + (f" (x{n})" if n > 1 else ""))
    card = _preflight(modules[2])
    backend = os.environ["CIRCUIT_SIM_BACKEND"]
    print(f"[slic] {len(built)} task(s) from {spec_path}; tech {tech_node}, card {card}, simulator {backend}")
    if a.check:
        print("[slic] spec OK (structures and action spaces built; nothing simulated or written)")
        return 0

    env = {k: os.environ[k] for k in ("CIRCUITOP_TECH_NODE", "CIRCUITOP_TECH_MODEL", "CIRCUIT_SIM_BACKEND", "NGSPICE_BIN",
                                      "NGSPICE_AC_FOLLOW_SPECTRE", "CIRCUITOP_TECH_ROOT") if k in os.environ}
    scratch_root = tempfile.mkdtemp(prefix="slic_tasks_", dir=a.scratch)
    try:
        if a.workers == 1:
            _worker_init(env, scratch_root)
            results = [_simulate(b["sim_netlist"]) for b in built]
        else:
            with ProcessPoolExecutor(max_workers=a.workers, initializer=_worker_init, initargs=(env, scratch_root)) as ex:
                results = list(ex.map(_simulate, [b["sim_netlist"] for b in built]))
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    rows = []
    for b, r in zip(built, results):
        ex_ = b["row"]["extra_info"]
        ex_["initial_state_class"] = r["class"]
        ex_["env_kwargs"]["task_data"]["initial_state"]["metrics"] = r["metrics"]
        m = r["metrics"]
        note = f"  [{r['error_type']}: {r['feedback'][:160]}]" if r["error_type"] else ""
        print(f"  {ex_['task_id']}: {r['class']:<7} gain={m['gain_dB']} dB power={m['power_uW']} uW "
              f"bw={m['bandwidth_MHz']} MHz pm={m['phase_margin_deg']} deg{note}")
        rows.append(b["row"])
    _write(rows, out)
    n = Counter(r["extra_info"]["initial_state_class"] for r in rows)
    print(f"[slic] wrote {len(rows)} task(s) to {out} (valid {n.get('valid', 0)}, invalid {n.get('invalid', 0)})")
    return 0
