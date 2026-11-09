import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import time
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- PSF operating-point parsing patterns ---
MODEL_STRUCT_RE = re.compile(r'^\s*"([^"]+)"\s+STRUCT\s*\(', re.MULTILINE)
PROP_DEF_RE = re.compile(
    r'^\s*"([^"]+)"\s+(FLOAT|INT)\s+(?:DOUBLE|LONG|BYTE)?\s*PROP\s*\(',
    re.IGNORECASE | re.MULTILINE
)
VALUE_SECTION_RE= re.compile(r'^\s*VALUE\b', re.IGNORECASE)
ENTRY_HDR_RE    = re.compile(r'"\s*([^"]+)\s*"\s*"\s*([^"]+)\s*"\s*\(', re.IGNORECASE)
VALUE_NUM_RE    = re.compile(r'(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)|(?:nan|inf|-inf)', re.IGNORECASE)

POWER_SOURCES_RE = re.compile(r"CIRCUITOP_POWER_SOURCES:\s*([A-Za-z0-9_ ]+)")


def tagged_power_sources(netlist_text):
    """Upper-case names of the sources whose |V*I| is the reported power, read from the testbench marker
    `// CIRCUITOP_POWER_SOURCES: VDD_ADM VSS_ADM` (one amplifier + its bias block). Raises ValueError when the
    marker is missing instead of summing every source."""
    m = POWER_SOURCES_RE.search(netlist_text or "")
    if m is None:
        raise ValueError("netlist has no `// CIRCUITOP_POWER_SOURCES:` marker; build the testbench with "
                         "circuit/data_pre/netlist_gen.py (it names the supply sources whose power is reported)")
    return {t.upper() for t in m.group(1).split()}


@dataclass
class Dev:
    name: str
    model: str
    props: Dict[str, float]


@dataclass
class _ExecutionResult:
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class SpectreInfrastructureError(RuntimeError):
    """Remote Spectre transport/configuration failure; escapes ``run_simulation`` so an outage is never
    scored as an invalid circuit."""


class SpectreRunner:
    def __init__(
        self,
        scratch_dir="/tmp/spectre_scratch",
        backend=None,
        api_url=None,
        api_token=None,
    ):
        self.scratch_dir = Path(scratch_dir).resolve()
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.backend = (
            backend or os.environ.get("CIRCUITOP_SPECTRE_BACKEND", "local")
        ).strip().lower()
        if self.backend not in {"local", "http"}:
            raise ValueError(
                "CIRCUITOP_SPECTRE_BACKEND must be 'local' or 'http', "
                f"got {self.backend!r}"
            )

        self.api_url = (
            api_url or os.environ.get("CIRCUITOP_SPECTRE_API_URL", "")
        ).strip().rstrip("/")
        self.api_token = (
            api_token or os.environ.get("CIRCUITOP_SPECTRE_API_TOKEN") or None
        )
        try:
            self.api_timeout_grace_seconds = int(
                os.environ.get(
                    "CIRCUITOP_SPECTRE_API_TIMEOUT_GRACE_SECONDS", "60"
                )
            )
            self.api_max_response_bytes = int(
                os.environ.get(
                    "CIRCUITOP_SPECTRE_API_MAX_RESPONSE_BYTES",
                    str(128 * 1024 * 1024),
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Spectre API timeout/response limits must be integers"
            ) from exc
        if self.api_timeout_grace_seconds < 0:
            raise ValueError(
                "CIRCUITOP_SPECTRE_API_TIMEOUT_GRACE_SECONDS must be >= 0"
            )
        if self.api_max_response_bytes <= 0:
            raise ValueError(
                "CIRCUITOP_SPECTRE_API_MAX_RESPONSE_BYTES must be > 0"
            )

        if self.backend == "http":
            if not self.api_url:
                raise SpectreInfrastructureError(
                    "CIRCUITOP_SPECTRE_API_URL is required for the HTTP backend"
                )
    # Fully differential testbenches have no CMFB, so DC can also settle in an all-off equilibrium. If the first run
    # does not amplify, retry with a `.nodeset` on the bias/output nodes at each ladder level (high first) and adopt
    # only an amplifying result; otherwise keep the first verdict.
    _FD_NS_LADDER = (0.45, 0.3, 0.2, 0.15)
    _FD_NS_NODES = ("vinp_fb", "vinn_fb", "vinp_fb_cm", "vinn_fb_cm",
                    "vinp_fb_dc", "vinn_fb_dc", "voutp_adm", "voutn_adm")

    @classmethod
    def _fd_nodeset_block(cls, netlist_text, level):
        nodes = [n for n in cls._FD_NS_NODES if n in netlist_text]
        return (netlist_text
                + "\nsimulator lang=spice\n.nodeset "
                + " ".join(f"v({n})={level}" for n in nodes)
                + "\nsimulator lang=spectre\n")
    # Licence-seat outages crash spectre at read-in on valid netlists; retry only spectre_crash (5 x 60 s). A bad
    # netlist fails deterministically on every attempt, so the retries mask nothing.
    _CRASH_RETRIES = 5
    _CRASH_BACKOFF_S = 60

    def run_simulation(self, netlist_content, timeout=900, threads=1):
        tagged_power_sources(netlist_content)   # raises before simulating when the testbench marker is missing
        metrics = self._run_simulation_once(netlist_content, timeout, threads)
        for attempt in range(self._CRASH_RETRIES):
            if metrics.get("error_type") != "spectre_crash":
                break
            print(f"[SpectreRunner] spectre_crash (attempt {attempt + 1}/"
                  f"{self._CRASH_RETRIES + 1}): {str(metrics.get('feedback'))[:120]} "
                  f"— retrying in {self._CRASH_BACKOFF_S}s", flush=True)
            time.sleep(self._CRASH_BACKOFF_S)
            metrics = self._run_simulation_once(netlist_content, timeout, threads)
        is_fd = ("voutp_adm" in netlist_content
                 and "voutn_adm" in netlist_content)
        gain = metrics.get("gain")
        needs_rescue = gain is None or gain != gain or gain <= 0
        if not (is_fd and needs_rescue):
            return metrics
        for level in self._FD_NS_LADDER:
            retry = self._run_simulation_once(
                self._fd_nodeset_block(netlist_content, level), timeout, threads)
            g = retry.get("gain")
            if g is not None and g == g and g > 0:
                retry["fd_nodeset_level"] = level
                return retry
        return metrics

    def _run_simulation_once(self, netlist_content, timeout=900, threads=1):
        """Runs Spectre and returns {gain, power, bw, op_data, valid}."""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        # mkdtemp: an atomic, unique directory per run, so parallel rollout workers never share or delete each other's
        # scratch directories.
        self.scratch_dir.mkdir(exist_ok=True, parents=True)
        work_dir = Path(
            tempfile.mkdtemp(prefix=f"run_{timestamp}_", dir=str(self.scratch_dir))
        ).resolve()
        run_id = work_dir.name.removeprefix("run_")
        
        netlist_path = (work_dir / "sim.scs").resolve()
        log_path = (work_dir / "spectre.log").resolve()
        psf_dir = (work_dir / "psf").resolve()

        # 1. Write Netlist
        with open(netlist_path, "w") as f:
            f.write(netlist_content)

        # 2. Prepare Command
        mt_flag = f"+mt={int(threads)}" if threads > 1 else ""
        command = (
            f"tcsh -c 'spectre -64 {netlist_path} +escchars "
            f"+log {log_path} -format psfascii -raw {psf_dir} "
            f"+lqtimeout {timeout} -maxw 5 -maxn 5 +logstatus {mt_flag}'"
        )
        

        metrics = {
            "valid": False, "gain": None, "power": None, "bandwidth_MHz": None, "phase_margin_deg": None,
            "differential_gain_db": None,
            "offset_v": None,
            "cmrr_db": None,
            "psrr_plus_db": None,
            "psrr_minus_db": None,
            "psrr_plus_dc_db": None,
            "psrr_plus_1mhz_db": None,
            "psrr_minus_dc_db": None,
            "psrr_minus_1mhz_db": None,
            "power_convention": None,
            "op_data": {}, "feedback": "", "error_type": None
        }

        try:
            if self.backend == "local":
                # Preserve the existing local execution behavior.
                process = subprocess.run(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, timeout=timeout + 30, cwd=str(work_dir)
                )
                execution = _ExecutionResult(
                    returncode=process.returncode,
                    stdout=process.stdout,
                    stderr=process.stderr,
                )
            else:
                execution = self._run_simulation_http(
                    netlist_content=netlist_content,
                    work_dir=work_dir,
                    timeout=timeout,
                    threads=threads,
                )

            if execution.timed_out:
                metrics["error_type"] = "timeout"
                metrics["feedback"] = f"Simulation timed out after {timeout}s"
                metrics["gain"] = np.nan
                metrics["power"] = np.nan
                metrics["bandwidth"] = np.nan
                metrics["phase_margin_deg"] = np.nan
                return metrics
            
            # 3. Check for Spectre Execution Errors
            if execution.returncode is None:
                raise SpectreInfrastructureError(
                    "Spectre API returned neither a return code nor a timeout"
                )
            if execution.returncode != 0:
                metrics["error_type"] = "spectre_crash"
                metrics["feedback"] = (
                    f"Return Code {execution.returncode}. "
                    f"Stderr: {execution.stderr.strip()}"
                )
                metrics["gain"] = np.nan
                metrics["power"] = np.nan
                metrics["bandwidth_MHz"] = None
                metrics["phase_margin_deg"] = np.nan
                if log_path.exists():
                    metrics["feedback"] += f"\nLog Tail: {self._read_log_tail(log_path)}"
                    # The tail often misses read-in errors (printed early); surface ERROR/license lines explicitly.
                    try:
                        _err_lines = [
                            l.strip() for l in
                            log_path.read_text(errors="replace").splitlines()
                            if ("ERROR" in l or "License" in l.title()
                                or "Unable" in l)][:6]
                        if _err_lines:
                            metrics["feedback"] += (
                                "\nError lines: " + " | ".join(_err_lines))
                        # CMI-2440: device W/L/area outside every bin of the model card. A design error, not an
                        # infrastructure failure, so it is not retried.
                        _bin = [l for l in _err_lines if "CMI-2440" in l]
                        if _bin:
                            metrics["error_type"] = "design_error"
                            metrics["feedback"] = (
                                "Design error: device geometry outside the model "
                                "card's bin range (Spectre CMI-2440): " + _bin[0][:300])
                    except Exception:
                        pass
                return metrics


            # 5. Parse Results
            try:
                self._parse_results(work_dir, netlist_path, metrics)
            except Exception as parse_err:
                metrics["error_type"] = "parsing_error"
                metrics["feedback"] = f"Error parsing PSF data: {str(parse_err)}"
                metrics["gain"] = np.nan
                metrics["power"] = np.nan
                metrics["bandwidth_MHz"] = None
                metrics["phase_margin_deg"] = np.nan
                return metrics

            if not metrics["valid"] and not metrics.get("error_type"):
                metrics["error_type"] = "missing_results"
                metrics["feedback"] = "Simulation returned no usable AC measurement."
            return metrics

        except SpectreInfrastructureError as exc:
            metrics["error_type"] = "infrastructure_error"
            metrics["feedback"] = str(exc)
            raise
        except subprocess.TimeoutExpired:
            metrics["error_type"] = "timeout"
            metrics["feedback"] = f"Simulation timed out after {timeout}s"
            metrics["gain"] = np.nan
            metrics["power"] = np.nan
            metrics["bandwidth"] = np.nan
            metrics["phase_margin_deg"] = np.nan
            return metrics
        except Exception as e:
            metrics["error_type"] = "python_exception"
            metrics["feedback"] = str(e)
            metrics["gain"] = np.nan
            metrics["power"] = np.nan
            metrics["bandwidth"] = np.nan
            metrics["phase_margin_deg"] = np.nan
            return metrics
        
        finally:
            if work_dir.exists():
                # Retry cleanup: NFS file locks can make rmtree fail transiently.
                import time
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(work_dir)
                        break
                    except (OSError, PermissionError) as e:
                        if attempt < max_attempts - 1:
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                        else:
                            # Last attempt failed - use ignore_errors to continue
                            try:
                                shutil.rmtree(work_dir, ignore_errors=True)
                            except:
                                pass  # Silently ignore - cleanup not critical

    def _run_simulation_http(
        self,
        *,
        netlist_content,
        work_dir,
        timeout,
        threads,
    ):
        endpoint = self.api_url
        if not endpoint.endswith("/v1/simulations"):
            endpoint = f"{endpoint}/v1/simulations"

        payload = {
            "netlist": netlist_content,
            "tech_node": os.environ.get("CIRCUITOP_TECH_NODE", "22nm"),
            "timeout_seconds": int(timeout),
            "threads": int(threads),
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/gzip",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        http_timeout = timeout + self.api_timeout_grace_seconds
        try:
            with urllib.request.urlopen(request, timeout=http_timeout) as response:
                api_version = response.headers.get("X-Spectre-API-Version")
                if api_version != "1":
                    raise SpectreInfrastructureError(
                        "Unsupported Spectre API version "
                        f"{api_version!r}; expected '1'"
                    )
                archive_bytes = response.read(self.api_max_response_bytes + 1)
        except urllib.error.HTTPError as exc:
            body = exc.read(64 * 1024).decode("utf-8", errors="replace")
            raise SpectreInfrastructureError(
                f"Spectre API returned HTTP {exc.code}: {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SpectreInfrastructureError(
                f"Spectre API request failed for {endpoint}: {exc}"
            ) from exc

        if len(archive_bytes) > self.api_max_response_bytes:
            raise SpectreInfrastructureError(
                "Spectre API response exceeds "
                f"{self.api_max_response_bytes} bytes"
            )

        manifest = self._extract_http_archive(
            archive_bytes=archive_bytes,
            work_dir=Path(work_dir),
        )
        if manifest.get("api_version") != "1":
            raise SpectreInfrastructureError(
                "Spectre artifact manifest has unsupported API version "
                f"{manifest.get('api_version')!r}"
            )

        timed_out = manifest.get("timed_out")
        if not isinstance(timed_out, bool):
            raise SpectreInfrastructureError(
                "Spectre artifact manifest timed_out must be a boolean"
            )
        return_code = manifest.get("return_code")
        if (
            return_code is not None
            and (
                not isinstance(return_code, int)
                or isinstance(return_code, bool)
            )
        ):
            raise SpectreInfrastructureError(
                "Spectre artifact manifest return_code must be an integer or null"
            )
        return _ExecutionResult(
            returncode=return_code,
            stdout=str(manifest.get("stdout", "")),
            stderr=str(manifest.get("stderr", "")),
            timed_out=timed_out,
        )

    def _extract_http_archive(self, *, archive_bytes, work_dir):
        allowed_exact = {"manifest.json", "spectre.log", "spectre.dc"}
        manifest = None
        total_uncompressed = 0

        try:
            archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz")
        except (tarfile.TarError, OSError) as exc:
            raise SpectreInfrastructureError(
                f"Invalid Spectre API artifact archive: {exc}"
            ) from exc

        with archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise SpectreInfrastructureError(
                    "Spectre API artifact archive contains duplicate paths"
                )

            for member in members:
                name = member.name
                if not (
                    name in allowed_exact
                    or (
                        name.startswith("psf/")
                        and name != "psf/"
                    )
                ):
                    raise SpectreInfrastructureError(
                        f"Unexpected path in Spectre API archive: {name!r}"
                    )
                if not member.isfile():
                    raise SpectreInfrastructureError(
                        f"Non-file entry in Spectre API archive: {name!r}"
                    )

                total_uncompressed += member.size
                if total_uncompressed > self.api_max_response_bytes:
                    raise SpectreInfrastructureError(
                        "Uncompressed Spectre API artifacts exceed "
                        f"{self.api_max_response_bytes} bytes"
                    )

                source = archive.extractfile(member)
                if source is None:
                    raise SpectreInfrastructureError(
                        f"Cannot read Spectre API artifact {name!r}"
                    )
                content = source.read()
                if len(content) != member.size:
                    raise SpectreInfrastructureError(
                        f"Truncated Spectre API artifact {name!r}"
                    )

                if name == "manifest.json":
                    try:
                        manifest = json.loads(content.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SpectreInfrastructureError(
                            "Invalid manifest.json in Spectre API archive"
                        ) from exc
                    continue

                destination = (work_dir / name).resolve()
                try:
                    destination.relative_to(work_dir.resolve())
                except ValueError as exc:
                    raise SpectreInfrastructureError(
                        f"Unsafe path in Spectre API archive: {name!r}"
                    ) from exc
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)

        if not isinstance(manifest, dict):
            raise SpectreInfrastructureError(
                "Spectre API artifact archive is missing manifest.json"
            )
        return manifest

    def _parse_results(self, work_dir, netlist_path, metrics):
        psf_dir = work_dir / "psf"
        ac_file = psf_dir / "ac.ac"
        dcop_file = psf_dir / "dcOpInfo.info"
        dc_file = work_dir / "spectre.dc"

        # A. Robust OP Parsing
        if dcop_file.exists():
            try:
                # Use the robust internal parser
                raw_devs = self._parse_dcop_robust(dcop_file)
                # Convert dataclass to simple dict for JSON compatibility
                metrics["op_data"] = {k: v.props for k, v in raw_devs.items()}
                
                # Power = sum of 'pwr' over the supply sources the testbench marker names
                tagged = tagged_power_sources(Path(netlist_path).read_text(errors="ignore"))
                supplies = [device for device in raw_devs.values() if device.model == 'vsource'
                            and device.name.split(':')[0].upper() in tagged]
                found = {device.name.split(':')[0].upper() for device in supplies}
                metrics["power_convention"] = "tagged:" + "+".join(sorted(tagged))
                if found != tagged:
                    metrics["feedback"] += f" tagged power sources missing in dcOpInfo: {sorted(tagged - found)}."
                    supplies = []
                if supplies and all(device.props.get('pwr') is not None for device in supplies):
                    metrics["power"] = abs(sum(device.props['pwr'] for device in supplies))
                
            except Exception as e:
                print(f"[Warn] OP Parse error: {e}")

        node_voltages = {}
        if dc_file.exists():
            try:
                node_voltages = self._parse_dc_file(dc_file)
            except Exception as e:
                print(f"[Warn] DC Parse error: {e}")
        # DC node voltages, returned with the metrics.
        metrics["node_voltages"] = node_voltages

        # B. AC Parsing (Gain/BW)
        if ac_file.exists():
            try:
                traces = self._parse_ac_traces(ac_file)
                analysis_kind = self._infer_analysis_kind(netlist_path)
                if analysis_kind == "simple":
                    self._populate_simple_metrics(traces, netlist_path, metrics)
                elif analysis_kind == "complex_single_ended":
                    self._populate_single_ended_ota_metrics(traces, node_voltages, metrics)
                elif analysis_kind == "complex_fully_differential":
                    self._populate_fully_differential_ota_metrics(traces, node_voltages, metrics)

                gain_for_validation = metrics.get("differential_gain_db")
                if gain_for_validation is None:
                    gain_for_validation = metrics.get("gain")
                if gain_for_validation is not None and np.isfinite(gain_for_validation) and gain_for_validation > 0:
                    metrics["valid"] = True
                else:
                    metrics["error_type"] = "design_error"
                    metrics["feedback"] = (
                        f"Design produces non-positive gain: {gain_for_validation:.2f} dB."
                        if gain_for_validation is not None and not np.isnan(gain_for_validation)
                        else "Design produces invalid gain."
                    )
            except Exception as e:
                print(f"[Warn] AC Parse error: {e}")

    # --- dcOpInfo parsing ---
    def _parse_dcop_robust(self, path: Path) -> Dict[str, Dev]:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        lines = txt.splitlines()
        model_props = self._build_model_props(txt)
        entries = self._stream_value_entries(lines)
        devs: Dict[str, Dev] = {}
        for inst, model, blob in entries:
            order = model_props.get(model, [])
            toks  = [t for t in VALUE_NUM_RE.findall(blob) if t.strip()]
            vals  = [self._to_float(t) for t in toks]
            props = {k: v for k, v in zip(order, vals)}
            
            devs[inst] = Dev(inst, model, props)
        return devs

    def _build_model_props(self, full_text: str) -> Dict[str, List[str]]:
        type_pos = full_text.find('TYPE')
        scan_text = full_text[type_pos:] if type_pos >= 0 else full_text
        model_props = {}
        
        for m in MODEL_STRUCT_RE.finditer(scan_text):
            model = m.group(1).strip()
            abs_start = (type_pos if type_pos >= 0 else 0) + m.end() - 1
            body, _ = self._grab_paren_block(full_text, abs_start)
            order = [p.group(1).strip().lower() for p in PROP_DEF_RE.finditer(body)]
            model_props[model] = order
        
        return model_props


    def _stream_value_entries(self, lines: List[str]) -> List[Tuple[str, str, str]]:
        """Extract VALUE entries with proper paren balancing."""
        entries = []
        i, n = 0, len(lines)
        
        # Find VALUE section
        while i < n and not VALUE_SECTION_RE.search(lines[i]): 
            i += 1
        if i >= n: 
            return entries
        i += 1
        
        while i < n:
            line = lines[i].strip()
            if not line or (line.startswith('"') and not ENTRY_HDR_RE.search(line)):
                i += 1
                continue
            if line.upper().startswith("END"): 
                break
                
            m = ENTRY_HDR_RE.search(line)
            if not m: 
                i += 1
                continue
            
            inst, model = m.group(1).strip(), m.group(2).strip()
            remaining = line[m.end():].strip()
            
            # Check if opening paren and values are on same line
            if remaining.startswith('('):
                paren_count, j = 1, m.end() + 1
                while j < len(line) and paren_count > 0:
                    if line[j] == '(': paren_count += 1
                    elif line[j] == ')': paren_count -= 1
                    j += 1
                if paren_count == 0:
                    # Complete entry on same line
                    values_text = line[m.end():j]
                    entries.append((inst, model, values_text))
                    i += 1
                    continue
            
            # Multi-line value collection with balanced paren counting
            value_lines = []
            paren_count = 1
            i += 1
            
            while i < n and paren_count > 0:
                line2 = lines[i]
                # Count parens in this line
                new_pc = paren_count + line2.count('(') - line2.count(')')
                
                if new_pc == 0:
                    # Found closing paren
                    end_idx = line2.find(')')
                    if end_idx >= 0:
                        value_lines.append(line2[:end_idx+1])
                    break
                else:
                    value_lines.append(line2)
                    paren_count = new_pc
                    i += 1
            
            values_text = " ".join(value_lines)
            entries.append((inst, model, values_text))
            i += 1
        
        return entries

    def _grab_paren_block(self, text: str, start_idx: int) -> Tuple[str, int]:
        depth, i, n = 1, start_idx + 1, len(text)
        out = []
        while i < n and depth > 0:
            ch = text[i]
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if depth > 0: out.append(ch)
            i += 1
        return ''.join(out), i

    def _to_float(self, tok: str) -> float:
        t = tok.strip().lower()
        if t == 'nan': return float('nan')
        if t == 'inf': return float('inf')
        return float(tok)

    def _read_log_tail(self, log_path, n=20):
        if not log_path.exists(): return "No log file."
        try:
            return "".join(log_path.read_text(errors='replace').splitlines()[-n:])
        except: return "Error reading log."

    # --- AC PARSING ---
    def _infer_analysis_kind_from_topology_names(self, text):
        instance_types = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue
            if not stripped.startswith(("X0", "XDUT_", "Xmain", "XMAIN")):
                continue
            match = re.search(r"\)\s+(\S+)", stripped)
            if match:
                instance_types.append(match.group(1).strip().lower())

        for inst_type in instance_types:
            if "fully_differential" in inst_type:
                return "complex_fully_differential", "topology_name"
        for inst_type in instance_types:
            if "single_ended" in inst_type:
                return "complex_single_ended", "topology_name"
        return None, None

    def _infer_analysis_kind(self, netlist_path):
        text = Path(netlist_path).read_text(errors='replace')
        marker = re.search(r'^\s*//\s*CIRCUITOP_ANALYSIS_KIND:\s*(\S+)\s*$', text, re.MULTILINE)
        if marker:
            kind = marker.group(1).strip()
            return kind
        if "XDUT_ADM" in text and "XDUT_ACM" in text:
            if "voutn_adm" in text and "voutp_adm" in text:
                kind = "complex_fully_differential"
            else:
                kind = "complex_single_ended"
            return kind

        topology_hint, source = self._infer_analysis_kind_from_topology_names(text)
        if topology_hint:
            return topology_hint

        message = (
            f"Unable to infer analysis kind for netlist '{netlist_path}'. "
            "Missing CIRCUITOP_ANALYSIS_KIND marker and no supported complex "
            "testbench/topology signature was found. Refusing to silently fall back to simple."
        )
        raise ValueError(message)

    def _find_simple_input_node(self, netlist_path):
        vin_node = None
        with open(netlist_path) as f:
            for line in f:
                if 'vsource' in line.lower() and 'mag=1' in line:
                    tokens = re.split(r'[()\s]+', line.strip())
                    if len(tokens)>=3:
                        vin_node=tokens[1]
                        break
        if not vin_node:
            raise ValueError("No AC source found")
        return vin_node.lower()

    def _parse_ac_traces(self, ac_file):
        traces = {}
        current_idx = -1
        content = ac_file.read_text(errors='replace')
        val_sec = re.search(r'VALUE(.*?)(END|$)', content, re.DOTALL | re.IGNORECASE)
        if not val_sec:
            return traces
        for raw_line in val_sec.group(1).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = re.match(r'^"([^"]+)"\s+(.+)$', line)
            if not m:
                continue
            key = m.group(1).lower()
            value_blob = m.group(2).strip()
            if key == "freq":
                val = float(value_blob.split()[0])
                traces.setdefault("freq", []).append(val)
                current_idx += 1
                for trace_key, vals in traces.items():
                    if trace_key != "freq" and len(vals) < current_idx:
                        vals.extend([0j] * (current_idx - len(vals)))
                continue
            if current_idx < 0:
                continue
            val = self._parse_complex_value(value_blob)
            vals = traces.setdefault(key, [])
            while len(vals) < current_idx:
                vals.append(0j)
            vals.append(val)
        parsed = {}
        for key, vals in traces.items():
            if key == "freq":
                parsed[key] = np.array(vals, dtype=float)
            else:
                parsed[key] = np.array(vals, dtype=complex)
        return parsed

    def _parse_complex_value(self, blob):
        blob = blob.strip()
        if blob.startswith("(") and blob.endswith(")"):
            parts = blob[1:-1].split()
            if len(parts) >= 2:
                return complex(float(parts[0]), float(parts[1]))
            if len(parts) == 1:
                return complex(float(parts[0]), 0.0)
        return complex(float(blob.split()[0]), 0.0)

    def _parse_dc_file(self, dc_file):
        node_vals = {}
        for raw_line in dc_file.read_text(errors='replace').splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            try:
                node_vals[key.lower()] = float(parts[1])
            except ValueError:
                continue
        return node_vals

    def _populate_simple_metrics(self, traces, netlist_path, metrics):
        vin_key = self._find_simple_input_node(netlist_path)
        freq = traces.get('freq', np.array([]))
        vin = traces.get(vin_key, np.array([]))
        vout = traces.get('vout', np.array([]))
        if len(freq) == 0 or len(vin) == 0 or len(vout) == 0 or np.any(np.abs(vin) == 0):
            return
        gain_mag = np.abs(vout) / np.abs(vin)
        gain_db = 20 * np.log10(gain_mag + 1e-30)
        metrics['gain'] = float(np.round(self._sample_gain_db(freq, gain_db, 1e3), 2))
        metrics['inband_flatness_db'] = self._calc_inband_flatness_db(gain_db)
        metrics['bandwidth_MHz'], metrics['bandwidth_kind'] = self._calc_bandwidth_mhz(freq, gain_db)
        metrics['phase_margin_deg'] = self._calc_phase_margin_from_transfer(freq, vout / (vin + 1e-30))
        metrics['phase_margin_status'] = self._pm_status(metrics)

    def _populate_single_ended_ota_metrics(self, traces, node_voltages, metrics):
        freq = traces.get('freq', np.array([]))
        signal_in = traces.get('signal_in', np.array([]))
        vout_adm = traces.get('vout_adm', np.array([]))
        if len(freq) == 0 or len(signal_in) == 0 or len(vout_adm) == 0:
            return
        adm_transfer = vout_adm / (signal_in + 1e-30)
        adm_gain_db = 20 * np.log10(np.abs(adm_transfer) + 1e-30)
        metrics['gain'] = float(np.round(self._sample_gain_db(freq, adm_gain_db, 1e3), 2))
        metrics['inband_flatness_db'] = self._calc_inband_flatness_db(adm_gain_db)
        metrics['differential_gain_db'] = metrics['gain']
        metrics['bandwidth_MHz'], metrics['bandwidth_kind'] = self._calc_bandwidth_mhz(freq, adm_gain_db)
        metrics['phase_margin_deg'] = self._calc_phase_margin_from_transfer(freq, adm_transfer)
        metrics['phase_margin_status'] = self._pm_status(metrics)

        cm_gain = traces.get('vout_cm', np.array([]))
        psrrp_gain = traces.get('vout_psrrp', np.array([]))
        psrrn_gain = traces.get('vout_psrrn', np.array([]))
        cm_db = None
        if len(cm_gain):
            cm_gain_db = 20 * np.log10(np.abs(cm_gain) + 1e-30)
            cm_db = self._sample_gain_db(freq, cm_gain_db, 1e3)
        psrrp_dc_db = self._db0(psrrp_gain[0]) if len(psrrp_gain) else None
        psrrn_dc_db = self._db0(psrrn_gain[0]) if len(psrrn_gain) else None
        psrrp_1m_db = self._db_at(freq, psrrp_gain, 1e6)
        psrrn_1m_db = self._db_at(freq, psrrn_gain, 1e6)
        if cm_db is not None:
            metrics['cmrr_db'] = float(np.round(metrics['gain'] - cm_db, 2))
        if psrrp_dc_db is not None:
            metrics['psrr_plus_dc_db'] = float(np.round(metrics['gain'] - psrrp_dc_db, 2))
        if psrrp_1m_db is not None:
            metrics['psrr_plus_1mhz_db'] = float(np.round(metrics['gain'] - psrrp_1m_db, 2))
        if psrrn_dc_db is not None:
            metrics['psrr_minus_dc_db'] = float(np.round(metrics['gain'] - psrrn_dc_db, 2))
        if psrrn_1m_db is not None:
            metrics['psrr_minus_1mhz_db'] = float(np.round(metrics['gain'] - psrrn_1m_db, 2))
        psrrp_candidates = [v for v in [metrics['psrr_plus_dc_db'], metrics['psrr_plus_1mhz_db']] if v is not None]
        psrrn_candidates = [v for v in [metrics['psrr_minus_dc_db'], metrics['psrr_minus_1mhz_db']] if v is not None]
        if psrrp_candidates:
            metrics['psrr_plus_db'] = float(np.round(min(psrrp_candidates), 2))
        if psrrn_candidates:
            metrics['psrr_minus_db'] = float(np.round(min(psrrn_candidates), 2))

        if 'vout_dc' in node_voltages:
            vcm = node_voltages.get('opin_dc', node_voltages.get('signal_in', node_voltages.get('opin')))
            if vcm is not None:
                metrics['offset_v'] = float(node_voltages['vout_dc'] - vcm)

    def _populate_fully_differential_ota_metrics(self, traces, node_voltages, metrics):
        freq = traces.get('freq', np.array([]))
        voutp_adm = traces.get('voutp_adm', np.array([]))
        voutn_adm = traces.get('voutn_adm', np.array([]))
        vinp_src = traces.get('vinp_src', np.array([]))
        vinn_src = traces.get('vinn_src', np.array([]))
        if len(freq) == 0 or len(voutp_adm) == 0 or len(voutn_adm) == 0 or len(vinp_src) == 0 or len(vinn_src) == 0:
            return
        adm_out = voutp_adm - voutn_adm
        adm_in = vinp_src - vinn_src
        adm_transfer = adm_out / (adm_in + 1e-30)
        adm_gain_db = 20 * np.log10(np.abs(adm_transfer) + 1e-30)
        metrics['gain'] = float(np.round(self._sample_gain_db(freq, adm_gain_db, 1e3), 2))
        metrics['inband_flatness_db'] = self._calc_inband_flatness_db(adm_gain_db)
        metrics['differential_gain_db'] = metrics['gain']
        metrics['bandwidth_MHz'], metrics['bandwidth_kind'] = self._calc_bandwidth_mhz(freq, adm_gain_db)
        metrics['phase_margin_deg'] = self._calc_phase_margin_from_transfer(freq, adm_transfer)
        metrics['phase_margin_status'] = self._pm_status(metrics)

        cm_out_p = traces.get('voutp_cm', np.array([]))
        cm_out_n = traces.get('voutn_cm', np.array([]))
        psrrp_out_p = traces.get('voutp_psrrp', np.array([]))
        psrrp_out_n = traces.get('voutn_psrrp', np.array([]))
        psrrn_out_p = traces.get('voutp_psrrn', np.array([]))
        psrrn_out_n = traces.get('voutn_psrrn', np.array([]))
        cm_vec = 0.5 * (cm_out_p + cm_out_n) if len(cm_out_p) and len(cm_out_n) else np.array([])
        cm_db = None
        if len(cm_vec):
            cm_gain_db = 20 * np.log10(np.abs(cm_vec) + 1e-30)
            cm_db = self._sample_gain_db(freq, cm_gain_db, 1e3)
        psrrp_vec = (psrrp_out_p - psrrp_out_n) if len(psrrp_out_p) and len(psrrp_out_n) else np.array([])
        psrrn_vec = (psrrn_out_p - psrrn_out_n) if len(psrrn_out_p) and len(psrrn_out_n) else np.array([])
        psrrp_dc_db = self._db0(psrrp_vec[0]) if len(psrrp_vec) else None
        psrrn_dc_db = self._db0(psrrn_vec[0]) if len(psrrn_vec) else None
        psrrp_1m_db = self._db_at(freq, psrrp_vec, 1e6) if len(psrrp_vec) else None
        psrrn_1m_db = self._db_at(freq, psrrn_vec, 1e6) if len(psrrn_vec) else None
        if cm_db is not None:
            metrics['cmrr_db'] = float(np.round(metrics['gain'] - cm_db, 2))
        if psrrp_dc_db is not None:
            metrics['psrr_plus_dc_db'] = float(np.round(metrics['gain'] - psrrp_dc_db, 2))
        if psrrp_1m_db is not None:
            metrics['psrr_plus_1mhz_db'] = float(np.round(metrics['gain'] - psrrp_1m_db, 2))
        if psrrn_dc_db is not None:
            metrics['psrr_minus_dc_db'] = float(np.round(metrics['gain'] - psrrn_dc_db, 2))
        if psrrn_1m_db is not None:
            metrics['psrr_minus_1mhz_db'] = float(np.round(metrics['gain'] - psrrn_1m_db, 2))
        psrrp_candidates = [v for v in [metrics['psrr_plus_dc_db'], metrics['psrr_plus_1mhz_db']] if v is not None]
        psrrn_candidates = [v for v in [metrics['psrr_minus_dc_db'], metrics['psrr_minus_1mhz_db']] if v is not None]
        if psrrp_candidates:
            metrics['psrr_plus_db'] = float(np.round(min(psrrp_candidates), 2))
        if psrrn_candidates:
            metrics['psrr_minus_db'] = float(np.round(min(psrrn_candidates), 2))

        if 'voutp_dc' in node_voltages and 'voutn_dc' in node_voltages:
            metrics['offset_v'] = float(node_voltages['voutp_dc'] - node_voltages['voutn_dc'])

    def _db0(self, val):
        mag = abs(val)
        if mag <= 0:
            return None
        return float(20 * np.log10(mag))

    def _db_at(self, freq, vec, target_freq_hz):
        if len(freq) == 0 or len(vec) == 0:
            return None
        idx = int(np.argmin(np.abs(freq - target_freq_hz)))
        return self._db0(vec[idx])

    def _calc_bw(self, freq, gain_db):
        if len(gain_db)==0: return 0.0
        peak = np.max(gain_db)
        idx = np.where(gain_db >= peak - 3.0)[0]
        if len(idx) < 2: return 0.0
        return float(freq[idx[-1]] - freq[idx[0]])

    def _calc_inband_flatness_db(self, gain_db):
        """In-band flatness (dB): max gain within the -3 dB band minus the first-point (1 Hz) gain.

        Recorded only; > 6 dB marks in-band peaking.
        """
        import numpy as _np
        g = _np.asarray(gain_db, dtype=float)
        if g.size < 3:
            return None
        lo = float(g[0])
        below = _np.where(g < lo - 3.0)[0]
        band = g if below.size == 0 else g[:max(int(below[0]), 1)]
        return float(_np.round(float(band.max()) - lo, 2))

    def _sample_gain_db(self, freq, gain_db, target_freq_hz):
        """Gain (dB) at target_freq_hz, linear in log-frequency between neighbouring sweep points.

        Outside the swept range the nearest end point is used.
        """
        if len(freq) == 0 or len(gain_db) == 0:
            return 0.0
        f = np.asarray(freq, dtype=float); g = np.asarray(gain_db, dtype=float)
        if len(f) >= 2 and f[0] <= target_freq_hz <= f[-1] and np.all(np.diff(f) > 0):
            return float(np.interp(np.log10(target_freq_hz), np.log10(f), g))
        idx = int(np.argmin(np.abs(f - target_freq_hz)))
        return float(g[idx])

    def _pm_status(self, metrics):
        """Phase-margin status:
        measured            -- a 0 dB crossing exists, PM is the measured value
        stable_no_crossing  -- no crossing but a clean in-band response (inband <= 6): unconditionally
                               stable over the sweep, the spec passes; no value is fabricated, PM stays None
        undefined           -- no crossing and in-band peaking above the gate (or inband missing): unreliable
        """
        if metrics.get('phase_margin_deg') is not None:
            return 'measured'
        ib = metrics.get('inband_flatness_db')
        # "stable" requires a live amplifier (gain > 0). For a dead circuit the whole curve sits
        # below 0 dB, "no crossing" is trivially true and inband references a dead curve -> undefined.
        g = metrics.get('differential_gain_db')
        if g is None:
            g = metrics.get('gain')
        try:
            alive = g is not None and float(g) == float(g) and float(g) > 0.0
        except (TypeError, ValueError):
            alive = False
        return 'stable_no_crossing' if (alive and ib is not None and ib <= 6.0) else 'undefined'

    def _calc_bandwidth_mhz(self, freq, gain_db):
        """-3 dB bandwidth in MHz, relative to the low-frequency gain, interpolated in log-frequency.

        Returns (freq_MHz, 'f3db'), or (None, None) when the low-frequency gain is < 3 dB or no crossing lies in the sweep.
        """
        if len(freq) < 2 or len(gain_db) < 2:
            return None, None
        lo = float(gain_db[0])
        if lo >= 3.0:
            level = lo - 3.0
            idx = np.where(np.asarray(gain_db) < level)[0]
            if len(idx) == 0:
                return None, None
            j = int(idx[0])
            if j == 0:
                return float(freq[0]) / 1e6, 'f3db'
            g1, g2 = gain_db[j-1], gain_db[j]
            f1, f2 = freq[j-1], freq[j]
            if g2 == g1:
                return float(f2) / 1e6, 'f3db'
            import math
            lf = math.log10(f1) + (level - g1) * (math.log10(f2) - math.log10(f1)) / (g2 - g1)
            return float(10 ** lf) / 1e6, 'f3db'
        return None, None

    def _calc_unity_gain_freq(self, freq, gain_db):
        if len(freq) < 2 or len(gain_db) < 2:
            return 0.0
        idx_below = np.where(gain_db <= 0.0)[0]
        if len(idx_below) == 0:
            # No 0 dB crossing: UGF undefined (rewards treat None as not met).
            return None
        if idx_below[0] == 0:
            return 0.0
        idx2 = idx_below[0]
        idx1 = idx2 - 1
        g1, g2 = gain_db[idx1], gain_db[idx2]
        f1, f2 = freq[idx1], freq[idx2]
        if g2 == g1:
            return float(f2)
        t = (0.0 - g1) / (g2 - g1)
        return float(f1 + t * (f2 - f1))

    def _calc_phase_margin_from_transfer(self, freq, h):
        if len(freq) < 2:
            return None
        gain_db = 20 * np.log10(np.abs(h) + 1e-30)

        # Unwrap phase for continuous curve
        phase_deg = np.degrees(np.unwrap(np.angle(h)))

        # Reference: the low-frequency phase; instability = 180 deg of additional lag.
        phase0 = float(phase_deg[0])

        # 2. Find the 0 dB gain crossover
        idx_below = np.where(gain_db <= 0.0)[0]
        if len(idx_below) == 0 or idx_below[0] == 0:
            return None  # no proper 0 dB crossing

        idx2 = idx_below[0]
        idx1 = idx2 - 1

        # 3. Linear interpolation of phase at 0 dB
        g1, g2 = gain_db[idx1], gain_db[idx2]
        p1, p2 = phase_deg[idx1], phase_deg[idx2]
        if g2 == g1:
            interp_phase = p2
        else:
            t = (0.0 - g1) / (g2 - g1)
            interp_phase = p1 + t * (p2 - p1)

        # 4. Critical phase = initial phase - 180 deg (covers in-phase and inverted starts).
        critical_phase = phase0 - 180.0
        phase_margin_raw = interp_phase - critical_phase
        if not np.isfinite(phase_margin_raw):
            return None
        # Not folded into [-180, 180): negative = unstable in unity feedback, > 180 = phase lead at the crossing.
        return float(np.round(phase_margin_raw, 2))
