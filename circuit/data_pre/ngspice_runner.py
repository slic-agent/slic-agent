#!/usr/bin/env python3
"""
NgspiceRunner — ngspice backend with the SpectreRunner.run_simulation interface.

Subclasses SpectreRunner and reuses its metric code; only simulation and extraction are ngspice-specific:

  netlist_content (Spectre syntax, from netlist_gen)
    → translate_netlist()          Spectre→SPICE text (subckt/devices/sources)
    → ngspice -b                   .op + .ac (sweep window taken from the netlist)
    → parse __ac__ (wrdata)        traces {node: complex array}
    → parse __op__.txt (print all) node voltages + v*#branch currents
    → parse __show__.txt (show all) per-MOSFET OP → op_data
    → SpectreRunner populate fns   gain/bandwidth/PM/CMRR/PSRR/offset
    → power = Σ |V·I| over the tagged supply sources

op_data uses Spectre naming ("X0.MM3"); MOSFET props carry both lowercase (vgs/vth/vds/vdsat/gm/gds/id) and
display-cased (Vgs/Vds/Vth/Vdsat/Ids/Gm) keys; resistors get Vdrop/I from OP node voltages.

Error types mirror SpectreRunner ('spectre_crash' included). Selected by CIRCUIT_SIM_BACKEND=ngspice (the default,
see circuitop/env.py); NGSPICE_BIN names the binary (default: ngspice on PATH).
"""
from __future__ import annotations

import os
import re
import shutil
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    from spectre_runner import SpectreRunner, tagged_power_sources
except ImportError:  # pragma: no cover — direct module use from repo root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spectre_runner import SpectreRunner, tagged_power_sources


# ── Spectre → SPICE translation ──────────────────────────────────────────────

_SI_FIX = [(re.compile(r"(\d)M\b"), r"\1meg")]  # Spectre mega → SPICE meg


def _fix_units(expr: str) -> str:
    for rx, rep in _SI_FIX:
        expr = rx.sub(rep, expr)
    return expr


def _conv_value(v: str) -> str:
    v = _fix_units(v.strip())
    if re.fullmatch(r"[\d.]+(?:[eE][+-]?\d+)?[fpnumkMGT]?(?:eg)?", v):
        return v
    return "{" + v + "}"


def convert_model_deck(scs_path: str) -> str:
    out = ["* converted from Spectre deck: " + scs_path]
    cur = None
    for raw in open(scs_path):
        s = raw.strip()
        if not s or s.startswith("*"):
            continue
        if s.startswith(("simulator", "library", "section", "endsection", "endlibrary")):
            continue
        m = re.match(r"model\s+(\w+)\s+bsim4\s+type\s*=\s*([np])", s)
        if m:
            cur = m.group(1)
            out.append(f".model {cur} {'nmos' if m.group(2)=='n' else 'pmos'} level=54")
            continue
        if s.startswith("+") and cur:
            out.append(s)
    return "\n".join(out) + "\n"


class NgspiceRunner(SpectreRunner):
    """ngspice backend with the SpectreRunner interface."""

    _TRACES_BY_KIND = {
        "simple": ["vout"],  # + input node resolved from the netlist
        "complex_single_ended": [
            "vout_adm", "vout_cm", "vout_psrrp", "vout_psrrn", "signal_in"],
        "complex_fully_differential": [
            "voutp_adm", "voutn_adm", "vinp_src", "vinn_src",
            "voutp_cm", "voutn_cm", "voutp_psrrp", "voutn_psrrp",
            "voutp_psrrn", "voutn_psrrn"],
    }

    def __init__(self, scratch_dir="/tmp/ngspice_scratch", ngspice_bin=None):
        super().__init__(scratch_dir=scratch_dir)
        self.ngspice_bin = ngspice_bin or os.environ.get("NGSPICE_BIN", "ngspice")
        if shutil.which(self.ngspice_bin) is None:   # fail when the runner is built, not at the first simulation
            raise RuntimeError(f"ngspice binary not found: {self.ngspice_bin!r} (install ngspice or set NGSPICE_BIN)")
        self._model_lib_cache: dict[str, Path] = {}

    # ── model lib (one conversion per tech deck) ────────────────────────────
    def _model_lib_for(self, scs_include: str) -> Path:
        cached = self._model_lib_cache.get(scs_include)
        if cached and cached.exists():
            return cached
        lib = self.scratch_dir / (Path(scs_include).stem + "_ng.lib")
        # Atomic publish (temp file + os.replace): workers sharing the scratch dir never read a half-written deck.
        text = convert_model_deck(scs_include)
        if not (lib.exists() and lib.stat().st_size == len(text.encode())):
            tmp = lib.with_name(f".{lib.name}.{os.getpid()}.tmp")
            tmp.write_text(text)
            os.replace(tmp, lib)
        self._model_lib_cache[scs_include] = lib
        return lib

    # ── translation ─────────────────────────────────────────────────────────
    def translate_netlist(self, spectre_text: str, trace_nodes: list[str]) -> tuple[str, list, dict]:
        """Returns (spice_text, resistors, includes) — resistors as
        (name, node_a, node_b, value_expr) for Vdrop/I reconstruction."""
        lines_out = [".param vdd=1.2"]
        seen_params = {"vdd"}
        # Take the AC sweep window from the Spectre analysis line (NGSPICE_AC_FOLLOW_SPECTRE=1, default) so both
        # backends share the first point, the reference for flatness, f3db and PM; =0 uses a fixed 10 Hz..10 GHz sweep.
        ac_start, ac_stop, ac_dec = "10", "10g", "100"
        if os.environ.get("NGSPICE_AC_FOLLOW_SPECTRE", "1") == "1":
            ac_start, ac_stop = "1", "500g"   # netlist_gen's window, used only if the deck has no ac line
            _m = re.search(r"^\s*ac\s+ac\s+(.*)$", spectre_text, re.M)
            if _m:
                _kv = dict(re.findall(r"(\w+)=(\S+)", _m.group(1)))
                ac_start = _kv.get("start", ac_start).lower()
                ac_stop = _kv.get("stop", ac_stop).lower()
                ac_dec = _kv.get("dec", ac_dec)
            # Spectre 'M' = mega but SPICE 'm' = milli; Spectre 'k'/'G' map unchanged
            ac_start, ac_stop = (v.replace("meg", "MEG").replace("m", "MEG") if v.endswith("m") else v for v in (ac_start, ac_stop))
        resistors: list = []
        model_lib = None
        # hierarchical device→model map (the show-all model row misaligns when ngspice wraps wide tables)
        subckt_devs: dict[str, dict[str, str]] = {}
        subckt_ports: dict[str, list] = {}
        subckt_dev_terms: dict[str, dict[str, tuple]] = {}
        inst_binds: dict[str, tuple] = {}  # XNAME -> (subckt, parent nets)
        inst_of: dict[str, str] = {}
        cur_subckt = None
        for raw in spectre_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r'include\s+"([^"]+\.scs)"', line)
            if m:
                model_lib = self._model_lib_for(m.group(1))
                continue
            if line.startswith("//"):
                lines_out.append("* " + line[2:])
                continue
            if line.startswith(("simulator ", "simulatorOptions", "dcOp ", "dcOpInfo",
                                "saveOptions", "global ")):
                continue
            if re.match(r"ac\s+ac\s", line):
                continue
            m = re.match(r"parameters\s+(.*)", line)
            if m:
                for k, v in re.findall(r"(\w+)\s*=\s*(\S+)", m.group(1)):
                    if k not in seen_params:
                        seen_params.add(k)
                        lines_out.append(f".param {k}={_fix_units(v)}")
                continue
            m = re.match(r"subckt\s+(.*)", line)
            if m:
                parts_sc = m.group(1).split()
                cur_subckt = parts_sc[0]
                subckt_ports[cur_subckt] = [x.lower() for x in parts_sc[1:]]
                subckt_devs.setdefault(cur_subckt, {})
                subckt_dev_terms.setdefault(cur_subckt, {})
                lines_out.append(".subckt " + m.group(1))
                continue
            if line == "ends" or line.startswith("ends "):
                cur_subckt = None
                lines_out.append(".ends")
                continue
            m = re.match(r"(\w+)\s*\(([^)]*)\)\s*(\w+)\s*(.*)", line)
            if m:
                name, nodes, prim, rest = m.group(1), m.group(2).split(), m.group(3), m.group(4)
                kv = dict(re.findall(r"(\w+)\s*=\s*(\S+)", rest))
                if prim in ("nch", "pch"):
                    if cur_subckt:
                        subckt_devs[cur_subckt][name.upper()] = prim
                        subckt_dev_terms[cur_subckt][name.upper()] = tuple(n.lower() for n in nodes[:4])
                    lines_out.append(f"{name} {' '.join(nodes)} {prim} "
                                     f"W={_conv_value(kv.get('w','1u'))} L={_conv_value(kv.get('l','45n'))}")
                elif prim == "resistor":
                    val = kv.get("r", "1k")
                    resistors.append((name, nodes[0], nodes[1], val))
                    lines_out.append(f"{name} {' '.join(nodes)} {_conv_value(val)}")
                elif prim == "capacitor":
                    lines_out.append(f"{name} {' '.join(nodes)} {_conv_value(kv.get('c','1p'))}")
                elif prim == "isource":
                    lines_out.append(f"{name} {' '.join(nodes)} DC {_conv_value(kv.get('dc','0'))}")
                elif prim == "vsource":
                    # AC magnitude may be a literal or a parameter (FD
                    # testbenches use mag=vdiff_ac)
                    ac = f" AC {_conv_value(kv['mag'])}" if "mag" in kv else ""
                    vname = name if name.upper().startswith("V") else "V" + name
                    lines_out.append(f"{vname} {' '.join(nodes)} DC {_conv_value(kv.get('dc','0'))}{ac}")
                elif prim == "vcvs":
                    ename = name if name.upper().startswith("E") else "E" + name
                    lines_out.append(f"{ename} {' '.join(nodes)} {_conv_value(kv.get('gain','1'))}")
                elif prim == "inductor":
                    lname = name if name.upper().startswith("L") else "L" + name
                    lines_out.append(f"{lname} {' '.join(nodes)} {_conv_value(kv.get('l','1u'))}")
                else:  # subckt instance
                    xname = name if name.upper().startswith("X") else "X" + name
                    inst_of[xname.upper()] = prim
                    inst_binds[xname.upper()] = (prim, [n.lower() for n in nodes])
                    lines_out.append(f"{xname} {' '.join(nodes)} {prim}")
                continue
            lines_out.append("* UNTRANSLATED: " + line)
        if model_lib is not None:
            lines_out.insert(0, f'.include "{model_lib}"')
        wr = " ".join(f"v({n})" for n in trace_nodes)
        # FD testbenches (no CMFB) have several DC equilibria; a .nodeset at vcm steers ngspice's OP to the amplifying
        # one instead of the latched/cutoff solution. FD-only.
        body = "\n".join(lines_out)
        fd_ns_nodes = [n for n in
                       ("vinp_fb", "vinn_fb", "voutp_adm", "voutn_adm",
                        "vinp_fb_cm", "vinn_fb_cm", "voutp_cm", "voutn_cm",
                        "vinp_fb_dc", "vinn_fb_dc", "voutp_dc", "voutn_dc")
                       if n in body]
        if "voutp_adm" in body and "voutn_adm" in body and fd_ns_nodes:
            lines_out.append(
                ".nodeset " + " ".join(f"v({n})={{vcm}}" for n in fd_ns_nodes))
        lines_out += [
            ".options temp=27 tnom=27 gmin=1e-12 reltol=1e-3 abstol=1e-12 vntol=1e-6",
            ".control",
            "op",
            "echo __OP_BEGIN__", "print all", "echo __OP_END__",
            "show all > __show__.txt",
            f"ac dec {ac_dec} {ac_start} {ac_stop}",
            f"wrdata __ac__ {wr}",
            "quit",
            ".endc",
            ".end",
        ]
        dev_models: dict[str, str] = {}
        dev_terms: dict[str, tuple] = {}
        for inst, (sub, parent_nets) in inst_binds.items():
            ports = subckt_ports.get(sub, [])
            bind = dict(zip(ports, parent_nets))
            for dev, model in subckt_devs.get(sub, {}).items():
                dev_models[f"{inst}.{dev}"] = model
                terms = subckt_dev_terms.get(sub, {}).get(dev)
                if terms:
                    dev_terms[f"{inst}.{dev}"] = tuple(
                        bind.get(t, f"{inst.lower()}.{t}") for t in terms)
        return ("translated from spectre by NgspiceRunner\n" + "\n".join(lines_out) + "\n",
                resistors, {"model_lib": str(model_lib) if model_lib else None,
                            "dev_models": dev_models, "dev_terms": dev_terms})

    # ── output parsing ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_wrdata(path: Path, trace_nodes: list[str]):
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        traces = {"freq": arr[:, 0]}
        # wrdata layout per requested vector: scale, real, imag
        for i, node in enumerate(trace_nodes):
            base = 3 * i
            traces[node] = arr[:, base + 1] + 1j * arr[:, base + 2]
        return traces

    @staticmethod
    def _parse_op_print(stdout: str):
        m = re.search(r"__OP_BEGIN__(.*?)__OP_END__", stdout, re.S)
        node_v, branches = {}, {}
        if not m:
            return node_v, branches
        for name, val in re.findall(r"^\s*(\S+)\s*=\s*([-\d.eE+]+)\s*$", m.group(1), re.M):
            if name.endswith("#branch"):
                branches[name[:-7].lower()] = float(val)
            else:
                node_v[name.lower()] = float(val)
        return node_v, branches

    @staticmethod
    def _parse_show_all(path: Path, dev_models: dict | None = None):
        """show all → {SpectreStyleName: {param: float}} for MOSFETs."""
        devs: dict[str, dict] = {}
        if not path.exists():
            return devs
        cols: list[str] = []
        for raw in open(path):
            line = raw.rstrip("\n")
            m = re.match(r"\s*device\s+(.*)", line)
            if m:
                cols = m.group(1).split()
                for c in cols:
                    devs.setdefault(c, {})
                continue
            if not cols:
                continue
            parts = line.split()
            if len(parts) == len(cols) + 1:
                key = parts[0]
                for c, v in zip(cols, parts[1:]):
                    try:
                        devs[c][key] = float(v)
                    except ValueError:
                        devs[c][key] = v
        out = {}
        for name, props in devs.items():
            # m.x0.mm3 → X0.MM3 ; m.x1.i0 handled the same way
            clean = re.sub(r"^m\.", "", name)
            clean = ".".join(p.upper() for p in clean.split("."))
            entry = dict(props)
            # Spectre reports PMOS values signed, ngspice `show` as magnitudes; _derive_signed_op_values fixes the
            # signs later from node voltages and terminal order.
            for lk, dk in (("vgs", "Vgs"), ("vds", "Vds"), ("vth", "Vth"),
                           ("vdsat", "Vdsat"), ("id", "Ids"), ("gm", "Gm")):
                if lk in entry:
                    entry[dk] = entry[lk]
            if "id" in entry:
                entry["ids"] = entry["id"]
            out[clean] = entry
        return out

    def _db0(self, val):
        mag = abs(val)
        if mag <= 0:
            # wrdata prints AC magnitudes below ~1e-30 as 0: rejection beyond the text resolution, not a missing
            # measurement, so floor it (missing traces still yield None upstream).
            mag = 1e-15
        return float(20 * np.log10(mag))

    @staticmethod
    def _derive_signed_op_values(op_data: dict, node_v: dict, dev_terms: dict) -> None:
        """Set vgs/vds from OP node voltages and terminal order (Spectre's definition), and give vth/vdsat/id the
        sign of the operating polarity (ngspice `show` mixes signed vds with magnitude vth)."""
        for dev, entry in op_data.items():
            terms = dev_terms.get(dev)
            if not terms or not isinstance(entry, dict) or "vgs" not in entry:
                continue
            d, g, sN = terms[0], terms[1], terms[2]
            vd, vg, vs = node_v.get(d), node_v.get(g), node_v.get(sN)
            if vd is None or vg is None or vs is None:
                continue
            vgs, vds = vg - vs, vd - vs
            entry["vgs"], entry["vds"] = vgs, vds
            sgn_g = 1.0 if vgs >= 0 else -1.0
            sgn_d = 1.0 if vds >= 0 else -1.0
            for k, sgn in (("vth", sgn_g), ("vdsat", sgn_g), ("id", sgn_d)):
                v = entry.get(k)
                if isinstance(v, float):
                    entry[k] = abs(v) * sgn
            for lk, dk in (("vgs", "Vgs"), ("vds", "Vds"), ("vth", "Vth"),
                           ("vdsat", "Vdsat"), ("id", "Ids")):
                if lk in entry:
                    entry[dk] = entry[lk]
            if "id" in entry:
                entry["ids"] = entry["id"]

    # ── main entry ──────────────────────────────────────────────────────────
    # FD tie-ladder: bias levels for the two-pass retry (see run_simulation). 0.6 V (= vcm on the 65nm deck) is
    # excluded: the latched solution sits there.
    _FD_TIE_LADDER = (0.15, 0.2, 0.3, 0.45)

    @staticmethod
    def _fd_tie(text: str, level: float) -> str:
        """Ideal-CMFB tie: 1T inductor from each FD input-bias node to a
        source at `level` V — DC forces input CM = level, AC open."""
        nodes = [n for n in ("vinp_fb", "vinn_fb", "vinp_fb_cm", "vinn_fb_cm",
                             "vinp_fb_dc", "vinn_fb_dc") if n in text]
        aid = [f"vcmaid_src vcmaid_n 0 dc {level}"] + [
            f"lcmaid_{i} {n} vcmaid_n 1e12" for i, n in enumerate(nodes)]
        return text.replace(".options temp=27",
                            "\n".join(aid) + "\n.options temp=27")

    @staticmethod
    def _fd_full_nodeset(text: str, volts: dict) -> str:
        """Replace the vcm-guess .nodeset with the full node-voltage vector
        harvested from a tie pass (original, unmodified circuit is solved)."""
        vv = {k: v for k, v in volts.items() if not k.startswith("vcmaid")}
        ns = ".nodeset " + " ".join(f"v({n})={v:.9g}" for n, v in sorted(vv.items()))
        if re.search(r"^\.nodeset .*$", text, flags=re.M):
            return re.sub(r"^\.nodeset .*$", ns, text, count=1, flags=re.M)
        return text.replace(".options temp=27", ns + "\n.options temp=27")

    def run_simulation(self, netlist_content, timeout=900, threads=1):
        tagged_power_sources(netlist_content)   # raises before simulating when the testbench marker is missing
        metrics, _ = self._run_once(netlist_content, timeout, threads)
        # FD multi-equilibrium rescue (no CMFB, so an all-off DC solution exists): for each tie level, solve with an
        # ideal-CMFB tie, then re-solve the ORIGINAL netlist seeded with that OP. An adopted result is a true
        # equilibrium of the unmodified circuit; with none amplifying, the first verdict stands.
        if (metrics.get("error_type") == "design_error"
                and "voutp_adm" in netlist_content
                and "voutn_adm" in netlist_content):
            for level in self._FD_TIE_LADDER:
                _, tie_node_v = self._run_once(
                    netlist_content, timeout, threads,
                    spice_transform=lambda t, _l=level: self._fd_tie(t, _l))
                if not tie_node_v:
                    continue
                released, _ = self._run_once(
                    netlist_content, timeout, threads,
                    spice_transform=lambda t, _v=tie_node_v: self._fd_full_nodeset(t, _v))
                g = released.get("differential_gain_db")
                if g is None:
                    g = released.get("gain")
                if g is not None and not np.isnan(g) and g > 0:
                    return released
            # no amplifying equilibrium found: keep the baseline result.
        return metrics

    def _run_once(self, netlist_content, timeout=900, threads=1,
                  spice_transform=None):
        # Full key template, as in SpectreRunner: the populate functions index several keys directly.
        metrics = {
            "valid": False, "gain": 0.0, "power": 0.0, "bandwidth": 0.0,
            "phase_margin_deg": None,
            "differential_gain_db": None,
            "offset_v": None,
            "cmrr_db": None,
            "psrr_plus_db": None,
            "psrr_minus_db": None,
            "psrr_plus_dc_db": None,
            "psrr_plus_1mhz_db": None,
            "psrr_minus_dc_db": None,
            "psrr_minus_1mhz_db": None,
            "op_data": {}, "feedback": "", "error_type": None,
        }
        work_dir = Path(tempfile.mkdtemp(prefix="ng_", dir=str(self.scratch_dir)))
        netlist_path = work_dir / "ref.scs"
        netlist_path.write_text(netlist_content)
        try:
            kind = self._infer_analysis_kind(netlist_path)
            trace_nodes = list(self._TRACES_BY_KIND.get(kind, ["vout"]))
            if kind == "simple":
                trace_nodes.append(self._find_simple_input_node(netlist_path))
            spice_text, resistors, meta = self.translate_netlist(netlist_content, trace_nodes)
            if spice_transform is not None:
                spice_text = spice_transform(spice_text)
            (work_dir / "sim.cir").write_text(spice_text)
            proc = subprocess.run([self.ngspice_bin, "-b", "sim.cir"], cwd=work_dir,
                                  capture_output=True, text=True, timeout=timeout)
            node_v, branches = self._parse_op_print(proc.stdout)
            metrics["op_data"] = self._parse_show_all(work_dir / "__show__.txt",
                                                      meta.get("dev_models"))
            self._derive_signed_op_values(metrics["op_data"], node_v,
                                          meta.get("dev_terms") or {})
            # power: |V * I| summed over the supply sources the testbench marker names (one amplifier + its bias block)
            tagged = tagged_power_sources(netlist_content)
            metrics["power_convention"] = "tagged:" + "+".join(sorted(tagged))
            want = {t.lower() for t in tagged}
            missing = sorted(want - set(branches))
            if missing:
                metrics["feedback"] += f" tagged power sources missing in the op print: {missing}."
            branches_for_power = {k: v for k, v in branches.items() if k in want} if not missing else {}
            total_p = 0.0
            for src, i_val in branches_for_power.items():
                # source node voltages: find "Vx" line nodes from the spice text
                m = re.search(rf"^{re.escape(src)}\s+(\S+)\s+(\S+)\s+DC",
                              spice_text, re.M | re.I)
                if not m:
                    continue
                vp = node_v.get(m.group(1).lower(), 0.0 if m.group(1) != "0" else 0.0)
                vn = node_v.get(m.group(2).lower(), 0.0)
                total_p += abs((vp - vn) * i_val)
            metrics["power"] = total_p if (tagged is None or branches_for_power) else None

            # resistor Vdrop / I for observation annotations
            for rname, na, nb, rval in resistors:
                va, vb = node_v.get(na.lower()), node_v.get(nb.lower())
                if va is None or vb is None:
                    continue
                # a resistor's current is not part of the observation (graph_serializer shows a resistor's value only)
                metrics["op_data"][rname.upper()] = {"model": "resistor", "Vdrop": va - vb, "I": None}

            ac_path = work_dir / "__ac__"
            if not ac_path.exists():
                ac_path = work_dir / "__ac__.data"
            if not ac_path.exists():
                metrics["error_type"] = "spectre_crash"
                # include stderr + exit code: a killed/aborted ngspice prints nothing after
                # "Circuit:" on stdout and the reason is only on stderr.
                metrics["feedback"] = ("ngspice produced no AC output. rc=" + str(proc.returncode)
                                       + " stdout tail: " + proc.stdout[-300:]
                                       + " | stderr tail: " + (proc.stderr or "")[-600:])
                return metrics, node_v
            traces = self._parse_wrdata(ac_path, trace_nodes)

            # DC aliases some populate paths expect
            for alias in ("voutp_dc", "voutn_dc", "vout_dc"):
                base = alias[:-3]
                if base in node_v:
                    node_v[alias] = node_v[base]

            if kind == "simple":
                # simple populate reads traces[vin_key] with the netlist-derived key
                self._populate_simple_metrics(traces, netlist_path, metrics)
            elif kind == "complex_single_ended":
                self._populate_single_ended_ota_metrics(traces, node_v, metrics)
            elif kind == "complex_fully_differential":
                self._populate_fully_differential_ota_metrics(traces, node_v, metrics)

            gain_val = metrics.get("differential_gain_db")
            if gain_val is None:
                gain_val = metrics.get("gain")
            if gain_val is not None and not np.isnan(gain_val) and gain_val > 0:
                metrics["valid"] = True
            else:
                metrics["error_type"] = "design_error"
                metrics["feedback"] = (
                    f"Design produces non-positive gain: {gain_val:.2f} dB."
                    if gain_val is not None and not np.isnan(gain_val)
                    else "Design produces invalid gain.")
            return metrics, node_v
        except subprocess.TimeoutExpired:
            metrics["error_type"] = "spectre_crash"
            metrics["feedback"] = f"ngspice timed out after {timeout}s"
            return metrics, {}
        except Exception as e:  # fail-informative, mirror SpectreRunner behavior
            metrics["error_type"] = "spectre_crash"
            metrics["feedback"] = f"ngspice backend error: {type(e).__name__}: {e}"
            return metrics, {}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
