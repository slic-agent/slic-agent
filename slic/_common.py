"""Shared helpers for the slic CLI: repo root, YAML configs, environment assembly, streaming subprocesses."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"[slic] config error: {msg}")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    data["__path__"] = str(p.resolve())
    return data


LOCAL_DIRS = ("data", "models", "outputs")   # repo-relative directories of downloads, checkpoints and run outputs


def resolve_path(value: str, base: Optional[Path] = None) -> str:
    """Expand ~ and env vars; make relative paths relative to the repo root (not the cwd). A value of the form
    <org>/<name> that does not exist under the repo root and does not start in one of the repo's data / models / outputs
    directories is a Hugging Face hub id (e.g. Qwen/Qwen3.5-4B) and is left alone."""
    v = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(v)
    if p.is_absolute():
        return v
    local = (base or REPO_ROOT) / p
    if not local.exists() and v.count("/") == 1 and p.parts[0] not in LOCAL_DIRS and not v.startswith("."):
        return v
    return str(local)


def as_env_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


def merge_env(*parts: Optional[Dict[str, Any]]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for part in parts:
        for k, v in (part or {}).items():
            if v is None:
                env.pop(k, None)
            else:
                env[str(k)] = as_env_value(v)
    return env


def print_plan(env: Dict[str, str], cmd: List[str]) -> None:
    print("[slic] environment:")
    for k in sorted(env):
        print(f"  {k}={env[k]}")
    print("[slic] command:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))


def stream_subprocess(cmd: List[str], env: Dict[str, str], line_hook: Optional[Callable[[str], None]] = None) -> int:
    """Run cmd with env merged over os.environ, echoing stdout/stderr line by line."""
    full_env = dict(os.environ)
    full_env.update(env)
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=full_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line_hook is not None:
                line_hook(line.rstrip("\n"))
    except KeyboardInterrupt:
        proc.terminate()
        raise
    return proc.wait()


def check_model_endpoint(base_url: str, model_name: str) -> None:
    """Fail fast unless the OpenAI-compatible endpoint answers and serves `model_name` (GET <base_url>/models): a wrong
    port can reach another server, whose 404 would otherwise surface only on the first model call."""
    import json as _json
    import urllib.request
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": "Bearer " + (os.environ.get("LOCAL_MODEL_API_KEY") or "EMPTY")}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as r:
            served = [m.get("id") for m in _json.loads(r.read().decode()).get("data", [])]
    except Exception as e:
        raise ConfigError(f"model endpoint {url} not reachable: {type(e).__name__}: {e} (is `vllm serve` running on that port?)")
    if model_name not in served:
        raise ConfigError(f"model endpoint {url} serves {served}, not {model_name!r}: wrong port, or set --served-model-name {model_name}")
