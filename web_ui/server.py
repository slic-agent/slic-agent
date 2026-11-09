"""
FastAPI server for CircuitOp web UI.
Provides API for models, topologies, tasks, and inference; serves the frontend.
"""
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(project_root))

from web_ui.topology_names import describe as describe_topology
from web_ui.streaming_inference import API_SAMPLING_DEFAULTS, protocol_sampling, run_streaming_inference

app = FastAPI(title="CircuitOp Web UI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config: dataset path and topology images folder
# 1) Environment variables (CIRCUITOP_DATASET_PATH, CIRCUITOP_TOPOLOGY_IMAGES_DIR)
# 2) Else web_ui/config.json (keys: dataset_path, topology_images_dir)
# 3) Else empty / default
def _load_config():
    out = {
        "dataset_path": "",
        "topology_images_dir": "",
        "local_model_base_url": "",
        "local_model_name": "",
    }
    config_file = Path(__file__).parent / "config.json"
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            out["dataset_path"] = data.get("dataset_path", "") or ""
            out["topology_images_dir"] = data.get("topology_images_dir", "") or ""
            out["local_model_base_url"] = data.get("local_model_base_url", "") or ""
            out["local_model_name"] = data.get("local_model_name", "") or ""
        except Exception:
            pass
    out["dataset_path"] = os.environ.get("CIRCUITOP_DATASET_PATH", "") or out["dataset_path"]
    out["topology_images_dir"] = os.environ.get("CIRCUITOP_TOPOLOGY_IMAGES_DIR", "") or out["topology_images_dir"]
    out["local_model_base_url"] = os.environ.get("CIRCUITOP_LOCAL_MODEL_BASE_URL", "") or out["local_model_base_url"]
    out["local_model_name"] = os.environ.get("CIRCUITOP_LOCAL_MODEL_NAME", "") or out["local_model_name"]
    return out

_config = _load_config()
DATASET_PATH = _config["dataset_path"]   # the 65 nm task set(s), os.pathsep-separated (slic serve --tasks)
# the node task sets: a directory with validation_<node>.parquet / test_ood_<node>.parquet (slic serve --node-tasks;
# `slic data --family nodes`); the cards come with `slic data --cards`
NODE_TASKS_DIR = os.environ.get("CIRCUITOP_NODE_TASKS_DIR", "")
OTHER_NODES = ("22nm", "32nm", "45nm", "90nm", "130nm")
# schematics: <dir>/<topology>/schematic.svg (`slic data --topologies` -> data/topologies) or <dir>/<topology>.svg|png
TOPOLOGY_IMAGES_DIR = _config["topology_images_dir"] or str(project_root / "data" / "topologies")
LOCAL_MODEL_BASE_URL = _config["local_model_base_url"]
LOCAL_MODEL_NAME = _config["local_model_name"] or "local"



def _present(x):
    """Drop the None entries parquet adds for struct keys that only other rows have (every topology's parameter names,
    say): the page shows and sends back only the task's own fields."""
    if isinstance(x, dict):
        return {k: _present(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_present(v) for v in x]
    return x


def _jsonable(x):
    """Task rows read from parquet hold numpy arrays / scalars, which FastAPI cannot encode: plain Python values."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):   # numpy array or scalar
        return _jsonable(x.tolist())
    return x


def _node_files(node: str) -> List[str]:
    if node == "65nm":
        return [p for p in DATASET_PATH.split(os.pathsep) if p.strip()]
    if node not in OTHER_NODES or not NODE_TASKS_DIR:
        return []
    d = Path(NODE_TASKS_DIR)
    return [str(f) for f in (d / f"validation_{node}.parquet", d / f"test_ood_{node}.parquet") if f.is_file()]


def _card_available(node: str) -> bool:
    from circuit.data_pre.netlist_gen import TECH_MODEL_MAP, TECH_ROOT
    return node in TECH_MODEL_MAP and (Path(TECH_ROOT) / TECH_MODEL_MAP[node]).is_file()


_DATASETS: Dict[str, Any] = {}


def _get_dataset(node: str = "65nm"):
    """(rows, sorted topology ids, {topology: held out}) of a node's task set; held out = appears only in test_ood files."""
    if node in _DATASETS:
        return _DATASETS[node]
    files = _node_files(node)
    if not files:
        raise HTTPException(status_code=404, detail=f"no task set for {node} (65 nm: slic serve --tasks; other nodes: "
                                                    "slic data --family nodes and slic serve --node-tasks data/nodes)")
    try:
        frames = []
        for f in files:   # one or several task parquets (e.g. test_iid + test_ood): their topologies are listed together
            df = pd.read_parquet(f)
            df["_file"] = Path(f).name
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {files}: {e}")
    where: Dict[str, set] = {}
    for e, fname in zip(df["extra_info"], df["_file"]):
        top = ((e or {}).get("env_kwargs") or {}).get("topology")
        if top:
            where.setdefault(top, set()).add(fname)
    held_out = {k: all("test_ood" in f for f in fs) for k, fs in where.items()}
    _DATASETS[node] = (df, sorted(where), held_out)
    return _DATASETS[node]


def _schematic_path(name: str) -> Optional[Path]:
    d = Path(TOPOLOGY_IMAGES_DIR)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    for p in (d / safe / "schematic.svg", d / safe / "schematic.png", d / f"{safe}.svg", d / f"{safe}.png"):
        if p.is_file():
            return p
    return None


def _task_rows(topology: str, node: str):
    df, topologies, _ = _get_dataset(node)
    if topology not in topologies:
        raise HTTPException(status_code=404, detail=f"Topology '{topology}' not found in the {node} task set")
    return df[df["extra_info"].apply(lambda x: (x or {}).get("env_kwargs", {}).get("topology") == topology)]


# --- API models ---
class InteractiveInferenceRequest(BaseModel):
    provider: str = "local"
    model_name: Optional[str] = None
    topology: str
    task_index: int = 0
    node: str = "65nm"                    # technology node: its task set and card (the task's starts are simulated at it)
    temperature: Optional[float] = None   # None: the evaluation protocol's (scripts/eval/run_eval.sh) for a local model,
    top_p: Optional[float] = None         # the API agent's default for an API model
    max_tokens: Optional[int] = None
    seed: Optional[int] = None            # None: every run samples afresh; set: a local vLLM server reproduces the episode
    max_steps: int = 32
    goal_specs: Optional[Dict[str, Any]] = None
    initial_params: Optional[Dict[str, Any]] = None


# --- Routes ---
@app.get("/api/models")
def list_models():
    """Return list of (provider, model_name) for the sidebar. Includes Local when config has local_model_base_url."""
    models = [
        {"provider": "openai", "model_name": "gpt-4o"},
        {"provider": "openai", "model_name": "gpt-4o-mini"},
        {"provider": "gemini", "model_name": "gemini-2.0-flash"},
        {"provider": "gemini", "model_name": "gemini-3-flash-preview"},
        {"provider": "deepseek", "model_name": "deepseek-chat"},
        {"provider": "deepseek", "model_name": "deepseek-reasoner"},
        {"provider": "grok", "model_name": "grok-4-1-fast-reasoning"},
        {"provider": "openrouter", "model_name": "xiaomi/mimo-v2-flash:free"},
        {"provider": "xiaomi", "model_name": "mimo-v2-flash"},
    ]
    if LOCAL_MODEL_BASE_URL:
        models.append({"provider": "local", "model_name": LOCAL_MODEL_NAME})
    for m in models:   # the sampling a run uses when the page leaves a field empty
        m["sampling"] = _local_sampling() if m["provider"] == "local" else dict(API_SAMPLING_DEFAULTS)
    return {"models": models}


_LOCAL_SAMPLING: Optional[Dict[str, Any]] = None


def _local_sampling() -> Dict[str, Any]:
    global _LOCAL_SAMPLING
    if _LOCAL_SAMPLING is None:
        _LOCAL_SAMPLING = protocol_sampling()
    return dict(_LOCAL_SAMPLING)


@app.get("/api/nodes")
def list_nodes():
    """The technology nodes with a task set (65 nm = slic serve --tasks), and whether each card is installed."""
    nodes = ([{"node": "65nm", "card": True}] if DATASET_PATH else []) + \
        [{"node": n, "card": _card_available(n)} for n in OTHER_NODES if _node_files(n)]
    return {"nodes": nodes}


@app.get("/api/topologies")
def list_topologies(node: str = Query("65nm")):
    """The topologies of a node's task set: ids, and per topology its readable family name, input device, whether it is
    held out (test_ood only) and whether a schematic is available."""
    if not _node_files(node):
        return {"topologies": [], "circuits": [], "message": "no task set: start the UI with slic serve --tasks <task parquet>"}
    _, topologies, held_out = _get_dataset(node)
    circuits = [dict(describe_topology(t), held_out=held_out[t], schematic=_schematic_path(t) is not None) for t in topologies]
    return {"topologies": topologies, "circuits": circuits}


@app.get("/api/task")
def get_task(
    topology: str = Query(..., description="Topology name"),
    task_index: int = Query(0, description="Index of task for this topology"),
    node: str = Query("65nm"),
):
    """Return one task (task_data, goal specs, initial metrics) for the given topology."""
    rows = _task_rows(topology, node)
    idx = task_index % len(rows)
    row = rows.iloc[idx]
    extra = row.get("extra_info") or {}
    env_kw = extra.get("env_kwargs") or {}
    task_data = env_kw.get("task_data") or {}
    initial = (task_data.get("initial_state") or {}).get("metrics") or {}
    return _jsonable({
        "task_data": _present(_jsonable(task_data)),
        "target": _present(_jsonable(dict(env_kw.get("target") or {}))),   # the row's goal (the page pre-fills its goal fields with it)
        "initial_metrics": _present(_jsonable(dict(initial))),
        "task_id": extra.get("task_id"),
        "task_index": idx,
        "n_tasks": len(rows),
        "topology": topology,
        "node": node,
    })


@app.get("/api/topology_image/{name}")
def get_topology_image(name: str):
    """The schematic of a topology (slic data --topologies; the page only asks when /api/topologies says one exists)."""
    p = _schematic_path(name)
    if p is None:
        raise HTTPException(status_code=404, detail=f"No schematic for '{name}' under {TOPOLOGY_IMAGES_DIR}")
    return FileResponse(p)


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_configured": bool(DATASET_PATH), "nodes": [n["node"] for n in list_nodes()["nodes"]]}


# --- Interactive streaming inference ---

@app.post("/api/interactive/stream")
def interactive_stream(req: InteractiveInferenceRequest):
    """SSE endpoint: streams step-by-step inference results as newline-delimited JSON."""
    if req.node != "65nm" and not _card_available(req.node):
        raise HTTPException(status_code=400, detail=f"no technology card for {req.node}: run slic data --cards")
    rows = _task_rows(req.topology, req.node)
    idx = req.task_index % len(rows)
    row = rows.iloc[idx]
    env_kw = (row.get("extra_info") or {}).get("env_kwargs") or {}
    task_data = copy.deepcopy(env_kw.get("task_data") or {})
    if not task_data:
        raise HTTPException(status_code=400, detail="Task has no task_data")
    if req.provider == "local" and not LOCAL_MODEL_BASE_URL:
        raise HTTPException(status_code=400, detail="Local model not configured: set local_model_base_url in web_ui/config.json")

    def event_stream():
        # the evaluation pipeline with the protocol of scripts/eval/run_eval.sh: the model sees exactly the `slic eval` prompts
        for chunk in run_streaming_inference(
            task={**env_kw, "task_data": task_data},
            provider=req.provider,
            model_name=req.model_name or (LOCAL_MODEL_NAME if req.provider == "local" else None),
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            seed=req.seed,
            max_steps=req.max_steps,
            tech_node=req.node,
            local_model_base_url=LOCAL_MODEL_BASE_URL if req.provider == "local" else None,
            local_model_name=LOCAL_MODEL_NAME if req.provider == "local" else None,
            goal_override=req.goal_specs,
            initial_params_override=req.initial_params,
        ):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Serve frontend ---
STATIC_DIR = Path(__file__).parent / "static"
INTERACTIVE_HTML = STATIC_DIR / "interactive.html"


@app.get("/")
def serve_interactive():
    if INTERACTIVE_HTML.exists():
        return FileResponse(INTERACTIVE_HTML)
    return PlainTextResponse("Interactive UI not found. Create web_ui/static/interactive.html", status_code=404)


@app.get("/static/{path:path}")
def serve_static(path: str):
    f = STATIC_DIR / path
    if f.exists() and f.is_file():
        return FileResponse(f)
    raise HTTPException(status_code=404)
