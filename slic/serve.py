"""`slic serve` — start the browser UI (FastAPI): run optimization episodes interactively, streamed turn by turn."""
from __future__ import annotations

import argparse
import os
from typing import List

from ._common import REPO_ROOT, check_model_endpoint, resolve_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic serve", description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)   # 8000 is vllm serve's default
    ap.add_argument("--tasks", nargs="+", help="task parquet(s) shown in the UI's topology / task picker, e.g. data/standard/test_iid.parquet "
                    "data/standard/test_ood.parquet (in-distribution and held-out topologies together)")
    ap.add_argument("--node-tasks", default=None, help="directory with the node task sets (validation_<node> / test_ood_<node>.parquet) "
                    "for the UI's technology-node switch (default: data/nodes if present; needs slic data --family nodes --cards)")
    ap.add_argument("--topologies", default=None, help="schematic directory (default: data/topologies, from slic data --topologies)")
    ap.add_argument("--model-url", help="OpenAI-compatible endpoint of your model, e.g. http://127.0.0.1:8000/v1 (vllm serve)")
    ap.add_argument("--model-name", help="the model name that endpoint serves (vllm serve --served-model-name)")
    a = ap.parse_args(argv)
    if a.tasks:
        os.environ["CIRCUITOP_DATASET_PATH"] = os.pathsep.join(resolve_path(p) for p in a.tasks)
    node_dir = a.node_tasks or (str(REPO_ROOT / "data" / "nodes") if (REPO_ROOT / "data" / "nodes").is_dir() else None)
    if node_dir:
        os.environ["CIRCUITOP_NODE_TASKS_DIR"] = resolve_path(node_dir)
    if a.topologies:
        os.environ["CIRCUITOP_TOPOLOGY_IMAGES_DIR"] = resolve_path(a.topologies)
    if a.model_url:   # the UI offers "local / <model-name>" in its model list
        check_model_endpoint(a.model_url, a.model_name or "local")
        print(f"[slic] model endpoint {a.model_url} serves {a.model_name or 'local'}")
        os.environ["CIRCUITOP_LOCAL_MODEL_BASE_URL"] = a.model_url
        os.environ["CIRCUITOP_LOCAL_MODEL_NAME"] = a.model_name or "local"
    os.environ.setdefault("CIRCUIT_SIM_BACKEND", "ngspice")
    os.environ.setdefault("NGSPICE_BIN", "ngspice")
    import uvicorn

    os.chdir(REPO_ROOT)
    uvicorn.run("web_ui.server:app", host=a.host, port=a.port, reload=False)
    return 0
