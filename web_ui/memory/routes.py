"""Experience database for the browser UI: record every run, replay recorded runs, and offer stored designs as warm starts.

`enable_memory(app, S, root)` is called by web_ui/server.py when `slic serve --memory-dir <root>` is given (S is the
web_ui.server module). Without the option none of this is loaded. It registers on the app:
  - POST /api/interactive/stream, the page's run endpoint, recording. It streams the same events as the plain endpoint,
    preceded by {"event": "memory", "run_id", "warm_start"}, and writes every event to the store as it passes. The request
    accepts an optional `warm_start: {run_id, step}`: that stored design becomes the starting sizes, and once the episode
    reports its starting metrics they must reproduce the stored ones, otherwise the episode stops with an error.
  - /api/memory/...: config, settings (how long unsaved runs are kept), runs (saved / unsaved), run detail, raw download,
    replay (SSE with the recorded timing), save / unsave, label and pin, and nearest (rank the saved designs for a goal).
  - /memory/memory.js (the run-page panel) and /database (the Database tab). /?replay=<run_id> replays a run;
    /?warm=<run_id>:<step> selects that run's task and fills its design as the start.

Every run is recorded; only the runs the user saves ("Save to database") are listed in the database and used for
retrieval. Unsaved runs are deleted a set number of hours after they end (default 24, or never; a setting of the store,
changed on the Database tab).

What gets recorded per run: the request; the init event (goal, tolerances, sampling, the simulated starting design); every
token/step/done/error event; and one state row per visited design.
  step -1  the starting design: its parameters and the metrics the environment re-simulated. Indexed (valid) only when the
           start is the task row's own or an unedited warm start that reproduced, because when the reset re-simulation fails
           the environment falls back to the row's stored metrics, which belong to the row's own parameters.
  step k   the design after turn k (resulting_params), with its metrics, validity, reward, score and the environment's
           four-spec verdict under the run's goal/tolerances.

No `from __future__ import annotations` here: FastAPI must see the real request-model class in the endpoint annotation.
"""
import asyncio
import hashlib
import json
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .ranking import Spec, rank
from .store import EpisodeStore, check_run_id

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "static"
STREAM_PATH = "/api/interactive/stream"
METRICS = (("gain_dB", "Gain", "dB"), ("power_uW", "Power", "uW"), ("bandwidth_MHz", "Bandwidth", "MHz"),
           ("phase_margin_deg", "Phase margin", "deg"))
# the environment's four-spec success (rewards.evaluate_constraint_specs) on the headline metrics; the gates it also applies
# (in-band flatness, phase-margin status, differential gain) are not in the event stream, so a ranked candidate's pass is an
# estimate. Scales (ranking.py): gain is in dB -> /20 (decades of voltage gain); power, bandwidth, PM are positive -> log10.
SPECS = (
    Spec("gain", "gain_dB", "min", "gain_min", scale="db"),
    Spec("power", "power_uW", "max", "power_max", scale="log"),
    Spec("bandwidth", "bandwidth_MHz", "min", "bandwidth_min", positive_goal_only=True, scale="log"),
    Spec("phase_margin", "phase_margin_deg", "window", "phase_margin_min_deg", "phase_margin_max_deg",
         positive_goal_only=True, scale="log"),
)
# operating conditions a stored design must share with the query (beyond topology | node): parameters the agent's actions do
# not change, so designs that differ in them answer a different problem
CONDITION_KEYS = ("vdd",)
REPRO_RTOL = 1e-6
RANKING_NOTE = ("Filtered to the same topology, node and vdd; distance D = sum of one-sided violations of the tolerated "
                "bounds in decades (gain dB/20; power, bandwidth, PM log10 ratio), so exceeding a minimum or staying under "
                "a maximum costs nothing; ties broken by robustness (smallest slack) then similarity. 'estimated' leaves "
                "out the environment's validity gates (in-band flatness, PM status, differential gain); 'exact' = the "
                "stored run had the same goal and tolerances. The new run's verdict is the environment's.")


# ---- helpers -----------------------------------------------------------------------------------------------------------
def sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, allow_nan=False)}\n\n"


def _finite_or_none(x):
    """JSON for the page: inf (a spec that cannot be evaluated) becomes None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite_or_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite_or_none(v) for v in x]
    return x


def _parse_chunk(chunk) -> Dict[str, Any]:
    s = chunk.decode() if isinstance(chunk, bytes) else chunk
    s = s.strip()
    if not s.startswith("data: ") or "\n" in s:
        raise RuntimeError(f"stream chunk is not one SSE event: {s[:200]!r}")
    return json.loads(s[6:])


def _metric(v):
    return None if v is None else float(v)


def _same_metric(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= REPRO_RTOL * max(abs(a), abs(b), 1e-12)


def _params(d: Optional[Dict[str, Any]]) -> Dict[str, str]:
    return {k: str(v).strip() for k, v in (d or {}).items() if v is not None and str(v).strip() != ""}


_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}


def parse_si(v) -> float:
    """A netlist number ('1.2', '449.3n', '1.1u', '2meg'; or a float) -> float. Anything else raises."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().lower()
    for suf in ("meg", "f", "p", "n", "u", "m", "k", "g", "t"):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * _SI[suf]
    return float(s)


def task_key(request: Dict[str, Any]) -> str:
    """Designs are only compared (and warm starts only allowed) within one topology | node."""
    top, node = request.get("topology"), request.get("node") or "65nm"
    if not top:
        raise ValueError("request has no topology")
    return f"{top}|{node}"


def normalize_goal(goal: Dict[str, Any]) -> Dict[str, Any]:
    """As CircuitopWorker.reset completes a goal: bandwidth 10 MHz and the PM window 30-90 deg when omitted."""
    g = {k: v for k, v in goal.items() if v is not None}
    g.setdefault("bandwidth_min", 10.0)
    g.setdefault("phase_margin_min_deg", 30.0)
    g.setdefault("phase_margin_max_deg", 90.0)
    return g


def apply_warm_start(request: Dict[str, Any], stored: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """The request with the stored design as its starting sizes; returns (request, edited). edited: the client sent sizes
    that differ from the stored design, so the reproduction check is skipped (and the start is not indexed)."""
    sent = request.get("initial_params")
    if sent is None:
        return {**request, "initial_params": dict(stored["state"])}, False
    return request, _params(sent) != _params(stored["state"])


def condition_filter(conditions: Optional[Dict[str, Any]]):
    """Stage-1 filter beyond the task key, from the query's operating conditions: candidate -> failed condition or None."""
    want = {}
    for k, v in (conditions or {}).items():
        if k not in CONDITION_KEYS:
            raise ValueError(f"unknown condition {k!r}; conditions are {list(CONDITION_KEYS)}")
        if v is None or str(v).strip() == "":
            continue
        want[k] = parse_si(v)

    def f(c):
        for k, x in want.items():
            got = c["state"].get(k)
            if got is None:
                return f"{k} missing"
            y = parse_si(got)
            if abs(x - y) > 1e-9 * max(abs(x), abs(y), 1e-12):
                return f"{k} differs"
        return None
    return f if want else None


def describe_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """The Database tab's topology / process / vdd columns."""
    ref = run.get("task_ref") or {}
    vdd = (run.get("start_state") or {}).get("vdd")
    return {"task": ref.get("topology") or run["task_key"].split("|")[0],
            "conditions": {"process": ref.get("node") or run["task_key"].split("|")[-1],
                           "vdd": None if vdd is None else f"{parse_si(vdd):g} V"}}


UI_CONFIG = {"metrics": [{"key": k, "label": l, "unit": u} for k, l, u in METRICS],
             "tolerance_keys": [s.name for s in SPECS], "condition_keys": list(CONDITION_KEYS),
             "task_column": "topology", "condition_columns": ["process", "vdd"]}


@dataclass
class _Episode:
    """What one recorded run has seen so far."""
    request: Dict[str, Any]
    warm_start: Optional[Dict[str, Any]] = None       # {run_id, step, edited}
    stored_state: Optional[Dict[str, Any]] = None     # the warm-start state row
    done: bool = False
    success: Optional[bool] = None
    best_step: Optional[int] = None
    error: Optional[str] = None


def on_event(evt: Dict[str, Any], rec, ep: _Episode) -> Optional[str]:
    """Record one event's states and fill `ep`. Returns a message to stop the episode on an invariant violation."""
    kind = evt.get("event")
    if kind == "init":
        rec.set_fields(goal=evt.get("goal_specs") or {}, tolerances=evt.get("tolerances") or {})
        metrics = {"gain_dB": _metric(evt.get("initial_gain_dB")), "power_uW": _metric(evt.get("initial_power_uW")),
                   "bandwidth_MHz": _metric(evt.get("initial_bandwidth_MHz")),
                   "phase_margin_deg": _metric(evt.get("initial_phase_margin_deg"))}
        params = evt.get("params") or {}
        ws, st = ep.warm_start, ep.stored_state
        if ws is not None and not ws["edited"]:
            if _params(params) != _params(st["state"]):
                return "warm start: the episode's starting parameters differ from the stored design"
            bad = [k for k, _, _ in METRICS if not _same_metric(metrics[k], st["metrics"].get(k))]
            if bad:
                return ("warm start did not reproduce the stored design: " +
                        ", ".join(f"{k} {metrics[k]} vs stored {st['metrics'].get(k)}" for k in bad))
        own_start = ep.request.get("initial_params") is None
        indexable = own_start or (ws is not None and not ws["edited"])
        rec.state(-1, params, metrics, valid=indexable, success=None)
    elif kind == "step":
        if "error" in evt:          # the model call failed: no design was produced
            ep.error = evt["error"]
            return None
        params = evt.get("params") or {}
        valid = bool(evt.get("valid"))
        if not params:
            if valid:
                raise RuntimeError(f"step {evt.get('step')}: valid turn without resulting parameters")
            return None
        metrics = {k: _metric(evt.get(k)) for k, _, _ in METRICS}
        rec.state(int(evt["step"]), params, metrics, valid=valid, success=bool(evt.get("success")),
                  reward=evt.get("reward"), score=evt.get("perf_score_post"))
    elif kind == "done":
        ep.done = True
        ep.success = bool(evt.get("success"))
        b = evt.get("best_step_idx")
        ep.best_step = None if b is None or int(b) < 0 else int(b)
    elif kind == "error":
        ep.error = evt.get("message") or "error"
    return None


def memory_sha256() -> str:
    """sha256 over this package's and the panel's source files, recorded with every run."""
    h = hashlib.sha256()
    files = sorted(p for p in HERE.glob("*.py")) + [STATIC / "memory.js"]
    for p in files:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


# ---- request models ----------------------------------------------------------------------------------------------------
class WarmStartRef(BaseModel):
    run_id: str
    step: int


class LabelRequest(BaseModel):
    label: Optional[str] = None
    pinned: Optional[bool] = None


class SaveRequest(BaseModel):
    label: Optional[str] = None


class SettingsRequest(BaseModel):
    unsaved_ttl_hours: Optional[float]                # required; null = never delete unsaved runs
    dry_run: bool = False                             # true: only report which unsaved runs the value would delete now


class NearestRequest(BaseModel):
    task: Dict[str, Any]                 # {topology, node}
    goal: Dict[str, Any]
    tolerances: Optional[Dict[str, float]] = None   # None: the protocol's
    conditions: Optional[Dict[str, Any]] = None     # operating conditions the candidates must share (vdd)
    weights: Optional[Dict[str, float]] = None      # per-spec weights w_j (default 1)
    k: int = 5


# ---- the app -----------------------------------------------------------------------------------------------------------
def enable_memory(app: FastAPI, S, root: str) -> EpisodeStore:
    """Open (or create) the store at `root` and register the recording stream, /api/memory/*, /memory/memory.js and
    /database on `app`. `S` is the web_ui.server module, which must not have registered the stream endpoint itself."""
    taken = {getattr(r, "path", None) for r in app.router.routes} & {STREAM_PATH, "/database"}
    if taken or any(str(getattr(r, "path", "")).startswith("/api/memory") for r in app.router.routes):
        raise RuntimeError(f"the app already serves {sorted(taken) or '/api/memory'}: with memory on, enable_memory registers them")
    store = EpisodeStore(root)
    rec_ = store.recover()
    if rec_["closed"]:
        print(f"[memory] closed {len(rec_['closed'])} stale run(s) left running by a dead server: {rec_['closed']}", flush=True)
    if rec_["live"]:
        print(f"[memory] left {len(rec_['live'])} run(s) alone: still being written, by another server sharing this "
              f"store: {rec_['live']}", flush=True)
    print(f"[memory] store {store.stats()}", flush=True)

    cache: Dict[str, Any] = {}

    def task_ref(request):
        t = S.get_task(topology=request["topology"], task_index=int(request.get("task_index") or 0),
                       node=request.get("node") or "65nm")
        return {"topology": request["topology"], "node": request.get("node") or "65nm", "task_index": t["task_index"],
                "task_id": t["task_id"], "n_tasks": t["n_tasks"], "dataset_path": S.DATASET_PATH}

    def model_info(request):
        out = {"provider": request.get("provider"), "model_name": request.get("model_name")}
        if request.get("provider") == "local":
            if "served" not in cache:   # the served model's /v1/models entry
                with urllib.request.urlopen(S.LOCAL_MODEL_BASE_URL.rstrip("/") + "/models", timeout=10) as r:
                    data = json.loads(r.read().decode())["data"]
                cache["served"] = {"models": [{"id": m.get("id"), "root": m.get("root"),
                                               "max_model_len": m.get("max_model_len")} for m in data]}
            out.update(base_url=S.LOCAL_MODEL_BASE_URL, model_name=S.LOCAL_MODEL_NAME, served=cache["served"])
        return out

    def code_info():
        if "code" not in cache:
            tree = Path(S.project_root)
            head = None
            if (tree / ".git").exists():
                head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True,
                                      check=True).stdout.strip()
            cache["code"] = {"tree": str(tree), "git_head": head, "memory_sha256": memory_sha256()}
        return cache["code"]

    def default_tolerances():
        if "tol" not in cache:
            cache["tol"] = dict(S.protocol()["tolerances"])
        return dict(cache["tol"])

    class RecordedRequest(S.InteractiveInferenceRequest):   # the page's request + an optional warm start
        warm_start: Optional[WarmStartRef] = None

    async def recorded_stream(req: RecordedRequest):
        request = req.model_dump(exclude={"warm_start"})
        tk = task_key(request)
        warm, stored = None, None
        if req.warm_start is not None:
            try:
                stored = store.get_state(check_run_id(req.warm_start.run_id), req.warm_start.step)
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=404, detail=f"warm start: {e}")
            if not stored["valid"]:
                raise HTTPException(status_code=400, detail="warm start: that state is not a valid design")
            if not store.get_run(stored["run_id"])["saved"]:
                raise HTTPException(status_code=400, detail="warm start: that run is not saved to the database")
            if stored["task_key"] != tk:
                raise HTTPException(status_code=400, detail=f"warm start: state is for {stored['task_key']}, run is {tk}")
            request, edited = apply_warm_start(request, stored)
            warm = {"run_id": stored["run_id"], "step": stored["step"], "edited": edited}
        resp = S.interactive_stream(S.InteractiveInferenceRequest(**request))   # validates (HTTPException) before recording
        store.purge_unsaved()
        rec = store.start_run(task_key=tk, task_ref=task_ref(request), request=request, model=model_info(request),
                              code=code_info(), warm_start=warm)
        ep = _Episode(request=request, warm_start=warm, stored_state=stored)

        async def gen():
            status, error = "aborted", None    # a client that disconnects mid-run leaves 'aborted' (all events so far kept)
            try:
                first = {"event": "memory", "run_id": rec.run_id, "warm_start": warm}
                rec.event(first)
                yield sse(first)
                async for chunk in resp.body_iterator:
                    evt = _parse_chunk(chunk)
                    rec.event(evt)
                    problem = on_event(evt, rec, ep)
                    if problem:                 # invariant violated (e.g. the warm start did not reproduce): stop the episode
                        err = {"event": "error", "message": f"memory: {problem}"}
                        rec.event(err)
                        status, error = "error", problem
                        yield sse(err)
                        return
                    yield chunk
                if ep.done:
                    status = "done"
                else:
                    status, error = "error", ep.error or "stream ended without a done event"
            except (asyncio.CancelledError, GeneratorExit):
                status, error = "aborted", "client disconnected"
                raise
            except Exception as e:              # recording or stream failure: tell the client, keep the run as an error
                status, error = "error", f"{type(e).__name__}: {e}"
                try:
                    yield sse({"event": "error", "message": f"memory: recording failed: {error}"})
                except Exception:
                    pass
                raise
            finally:
                rec.finish(status, error=error, success=ep.success, best_step=ep.best_step)
                aclose = getattr(resp.body_iterator, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.post(STREAM_PATH)(recorded_stream)

    @app.get("/api/memory/config")
    def memory_config():
        store.purge_unsaved()
        return {"stats": store.stats(), "ui": UI_CONFIG, "database_url": "/database", "specs": [s.__dict__ for s in SPECS]}

    @app.get("/api/memory/settings")
    def get_settings():
        return {"unsaved_ttl_hours": store.unsaved_ttl_hours}

    @app.put("/api/memory/settings")
    def put_settings(body: SettingsRequest):
        try:
            would = store.purgeable(body.unsaved_ttl_hours)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if body.dry_run:
            return {"unsaved_ttl_hours": body.unsaved_ttl_hours, "applied": False, "would_delete": would}
        store.set_unsaved_ttl_hours(body.unsaved_ttl_hours)
        return {"unsaved_ttl_hours": store.unsaved_ttl_hours, "applied": True, "deleted": store.purge_unsaved()}

    @app.get("/api/memory/runs")
    def memory_runs(task_key: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=1000),
                    saved: Optional[bool] = Query(None, description="true = the database, false = unsaved runs")):
        store.purge_unsaved()
        runs = store.list_runs(task_key=task_key, limit=limit, saved=saved)
        for r in runs:
            r.update(describe_run(r))
        return {"runs": runs}

    def _run_or_404(run_id: str):
        try:
            return store.get_run(check_run_id(run_id))
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/memory/runs/{run_id}")
    def memory_run(run_id: str):
        run = _run_or_404(run_id)
        return {"run": run, "states": store.get_states(run_id)}

    @app.get("/api/memory/runs/{run_id}/raw")
    def memory_raw(run_id: str):
        _run_or_404(run_id)
        p = store.raw_path(run_id, verify=True)
        return FileResponse(p, media_type="application/gzip", filename=p.name)

    @app.get("/api/memory/runs/{run_id}/replay")
    async def memory_replay(run_id: str, speed: float = Query(1.0, ge=0.0, le=1000.0),
                            max_gap: float = Query(3.0, ge=0.0, le=600.0)):
        """The recorded events with the recorded timing / speed (0 = at once); a pause longer than max_gap seconds
        (after scaling) is shortened to max_gap."""
        _run_or_404(run_id)
        events = list(store.iter_raw(run_id))   # sha-verified before the first byte is sent

        async def gen():
            yield sse({"event": "memory", "replay_of": run_id})
            prev = 0.0
            for t, e in events:
                if speed > 0:
                    await asyncio.sleep(min(max(0.0, t - prev) / speed, max_gap))
                prev = t
                if e.get("event") == "memory":
                    continue
                yield sse(e)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/memory/runs/{run_id}/label")
    def memory_label(run_id: str, body: LabelRequest):
        _run_or_404(run_id)
        try:
            store.label(run_id, label=body.label, pinned=body.pinned)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"run": store.get_run(run_id)}

    @app.post("/api/memory/runs/{run_id}/save")
    def memory_save(run_id: str, body: SaveRequest):
        _run_or_404(run_id)
        try:
            store.save(run_id, label=body.label)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"run": store.get_run(run_id)}

    @app.post("/api/memory/runs/{run_id}/unsave")
    def memory_unsave(run_id: str):
        _run_or_404(run_id)
        store.unsave(run_id)
        return {"run": store.get_run(run_id)}

    @app.post("/api/memory/nearest")
    def memory_nearest(body: NearestRequest):
        if not 1 <= body.k <= 50:
            raise HTTPException(status_code=400, detail="k must be 1..50")
        tk = task_key(body.task)
        tol = default_tolerances() if body.tolerances is None else {**default_tolerances(), **body.tolerances}
        cands = store.candidate_states(tk)
        for c in cands:   # both goals completed the same way, so an "exact" verdict compares like with like
            c["run_goal"] = normalize_goal(c["run_goal"]) if c["run_goal"] is not None else None
        try:
            res = rank(SPECS, cands, normalize_goal(body.goal), tol, k=body.k, weights=body.weights,
                       condition_filter=condition_filter(body.conditions))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _finite_or_none({"task_key": tk, "tolerances": tol, "conditions": body.conditions or {},
                                "goal": normalize_goal(body.goal), **res, "note": RANKING_NOTE})

    @app.get("/memory/memory.js")
    def memory_js():
        return FileResponse(STATIC / "memory.js", media_type="application/javascript")

    @app.get("/database")
    def database_page():
        return FileResponse(STATIC / "database.html", media_type="text/html")

    return store
