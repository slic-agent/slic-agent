"""Episode store: every deployed run's raw event stream, plus an index of the states it visited.

The meaning of an event (which states a run visited, its verdict) comes from web_ui/memory/routes.py; the store keeps:

  <root>/memory.sqlite   runs   one row per episode: request, goal, tolerances, model/code provenance, status, verdict,
                                and the raw file's path, sha256 and size
                         states one row per visited state (step -1 = the initial design, then every turn): the state
                                (e.g. device parameters), measured metrics, validity, the episode's own verdict.
                                This table is the retrieval index (web_ui/memory/ranking.py ranks it).
  <root>/raw/<run_id>.jsonl.gz   the raw stream. Line 1 is {"header": run row at start}; then one {"t": seconds since start,
                                 "e": event} per event, exactly as the client received it (tokens included), so a replay
                                 reproduces the run.

Saving is the user's choice. Every live run is staged: it is recorded in full while it runs, so a run that turns out
well can still be kept after it ends. It stays *unsaved* (saved=0) until the user saves it. Only saved runs are listed in
the database and indexed for retrieval. Unsaved runs are purged once they have been finished for `unsaved_ttl_hours`, a
setting kept in the store itself (meta table; default 24, or never), changed from the Database page.

While a run is live, its raw stream goes to raw/<run_id>.jsonl, with each event flushed. At the end the stream is
gzipped, hashed and recorded. A server crash leaves status 'running' and the .jsonl; `recover` closes such runs as
'interrupted' and keeps every event written so far. A raw file whose sha256 no longer matches its row is an error and is
never replayed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

SCHEMA_VERSION = 2   # 2: runs.saved / saved_utc (save-to-database); v1 stores are migrated with every run unsaved
RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")
STATUSES = ("running", "done", "error", "aborted", "interrupted")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  domain TEXT NOT NULL,
  task_key TEXT NOT NULL,          -- what states are comparable within (circuitop: topology|node)
  task_ref TEXT NOT NULL,          -- JSON: the task's identity (circuitop: topology, node, task_index, task_id)
  goal TEXT,                       -- JSON: the goal the episode ran with (from the init event)
  tolerances TEXT,                 -- JSON
  request TEXT NOT NULL,           -- JSON: the client's request, verbatim
  model TEXT NOT NULL,             -- JSON: model provenance
  code TEXT NOT NULL,              -- JSON: code/runtime provenance
  warm_start TEXT,                 -- JSON: {run_id, step, edited} when started from a stored state
  created_utc TEXT NOT NULL,
  finished_utc TEXT,
  status TEXT NOT NULL,
  error TEXT,
  success INTEGER,                 -- the episode's own verdict (the environment's), NULL until done
  best_step INTEGER,
  n_steps INTEGER NOT NULL DEFAULT 0,
  n_events INTEGER NOT NULL DEFAULT 0,
  raw_path TEXT NOT NULL,          -- relative to the store root
  raw_sha256 TEXT,
  raw_bytes INTEGER,
  label TEXT NOT NULL DEFAULT '',
  pinned INTEGER NOT NULL DEFAULT 0,
  saved INTEGER NOT NULL DEFAULT 0,  -- 1 = the user saved it to the database (listed + indexed); 0 = staged, purged after the TTL
  saved_utc TEXT
);
CREATE INDEX IF NOT EXISTS runs_task ON runs (task_key, created_utc);
CREATE TABLE IF NOT EXISTS states (
  run_id TEXT NOT NULL REFERENCES runs (run_id),
  step INTEGER NOT NULL,           -- -1 = initial design
  task_key TEXT NOT NULL,
  valid INTEGER NOT NULL,
  success INTEGER,                 -- verdict under the run's own goal/tolerances (environment), NULL for the initial design
  state TEXT NOT NULL,             -- JSON (circuitop: device parameters)
  state_sha256 TEXT NOT NULL,      -- of the canonical state JSON: dedup key
  metrics TEXT NOT NULL,           -- JSON (None = not measurable)
  reward REAL,
  score REAL,
  PRIMARY KEY (run_id, step)
);
CREATE INDEX IF NOT EXISTS states_task ON states (task_key, valid);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def canonical(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise ValueError(f"not a run id: {run_id!r}")
    return run_id


class RunRecorder:
    """Writes one live episode. Usage: rec = store.start_run(...); rec.event(e) per event; rec.state(...) per visited state;
    rec.finish(status, ...). Errors propagate: a run that cannot be recorded must not look recorded."""

    def __init__(self, store: "EpisodeStore", run_id: str, header: Dict[str, Any]):
        self.store, self.run_id = store, run_id
        self.t0 = time.monotonic()
        self.n_events = 0
        self.n_steps = 0
        self.closed = False
        self.live_path = store.root / "raw" / f"{run_id}.jsonl"
        self._f = open(self.live_path, "x", encoding="utf-8")
        self._write({"header": header})

    def _write(self, obj: Dict[str, Any]) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
        self._f.flush()

    def event(self, e: Dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError(f"run {self.run_id}: event after finish")
        self._write({"t": round(time.monotonic() - self.t0, 4), "e": e})
        self.n_events += 1

    def state(self, step: int, state: Dict[str, Any], metrics: Dict[str, Any], valid: bool, success: Optional[bool],
              reward: Optional[float] = None, score: Optional[float] = None) -> None:
        self.store._insert_state(self.run_id, step, state, metrics, valid, success, reward, score)
        if step >= 0:
            self.n_steps += 1

    def set_fields(self, **fields: Any) -> None:
        self.store._update_run(self.run_id, **fields)

    def finish(self, status: str, error: Optional[str] = None, success: Optional[bool] = None,
               best_step: Optional[int] = None) -> None:
        if self.closed:
            return
        self.closed = True
        self._f.close()
        self.store._close_raw(self.run_id, self.live_path, status=status, error=error, success=success, best_step=best_step,
                              n_steps=self.n_steps, n_events=self.n_events)


class EpisodeStore:
    DEFAULT_UNSAVED_TTL_HOURS = 24.0
    DOMAIN = "circuitop"     # recorded in every row and in meta; a store written for another domain is refused

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()
        self.domain = domain = self.DOMAIN
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.root / "memory.sqlite"), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(_SCHEMA)
            row = self._db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                self._db.execute("INSERT INTO meta VALUES ('schema_version', ?), ('domain', ?)", (str(SCHEMA_VERSION), domain))
            else:
                if int(row["value"]) == 1:   # v1 -> v2: add the save columns; v1 runs (recorded before saving was a choice) stay unsaved
                    cols = {r["name"] for r in self._db.execute("PRAGMA table_info(runs)")}
                    if "saved" not in cols:
                        self._db.execute("ALTER TABLE runs ADD COLUMN saved INTEGER NOT NULL DEFAULT 0")
                        self._db.execute("ALTER TABLE runs ADD COLUMN saved_utc TEXT")
                    self._db.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))
                elif int(row["value"]) != SCHEMA_VERSION:
                    raise RuntimeError(f"{self.root}: schema {row['value']}, this code reads {SCHEMA_VERSION}")
                d = self._db.execute("SELECT value FROM meta WHERE key='domain'").fetchone()["value"]
                if d != domain:
                    raise RuntimeError(f"{self.root} is a {d!r} store, not {domain!r}")
            self._db.execute("INSERT OR IGNORE INTO meta VALUES ('unsaved_ttl_hours', ?)", (repr(self.DEFAULT_UNSAVED_TTL_HOURS),))
            self.unsaved_ttl_hours   # parses the stored value; a corrupt one raises here, not at the first purge

    # ---- settings ------------------------------------------------------------------------------------------------
    @property
    def unsaved_ttl_hours(self) -> Optional[float]:
        """Hours after which an unsaved, finished run is deleted; None = never."""
        with self._lock:
            v = self._db.execute("SELECT value FROM meta WHERE key='unsaved_ttl_hours'").fetchone()["value"]
        return None if v == "never" else self._check_ttl(float(v))

    @staticmethod
    def _check_ttl(hours: Optional[float]) -> Optional[float]:
        if hours is None:
            return None
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or not hours > 0:
            raise ValueError(f"unsaved_ttl_hours={hours!r}: a number of hours > 0, or None (never delete)")
        return float(hours)

    def set_unsaved_ttl_hours(self, hours: Optional[float]) -> None:
        h = self._check_ttl(hours)
        with self._lock:
            self._db.execute("UPDATE meta SET value=? WHERE key='unsaved_ttl_hours'", ("never" if h is None else repr(h),))

    def purgeable(self, hours: Any = "setting") -> List[str]:
        """Unsaved, finished runs older than `hours` ("setting" = the current setting; None = never, so none)."""
        h = self.unsaved_ttl_hours if hours == "setting" else self._check_ttl(hours)
        if h is None:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            rows = self._db.execute("SELECT run_id FROM runs WHERE saved=0 AND status != 'running' AND finished_utc < ? "
                                    "ORDER BY run_id", (cutoff,)).fetchall()
        return [r["run_id"] for r in rows]

    # ---- writing -------------------------------------------------------------------------------------------------
    def start_run(self, task_key: str, task_ref: Dict[str, Any], request: Dict[str, Any], model: Dict[str, Any],
                  code: Dict[str, Any], warm_start: Optional[Dict[str, Any]] = None) -> RunRecorder:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        row = dict(run_id=run_id, domain=self.domain, task_key=task_key, task_ref=canonical(task_ref),
                   request=canonical(request), model=canonical(model), code=canonical(code),
                   warm_start=canonical(warm_start) if warm_start else None, created_utc=utc_now(), status="running",
                   raw_path=f"raw/{run_id}.jsonl.gz")
        with self._lock:
            cols = ",".join(row)
            self._db.execute(f"INSERT INTO runs ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        header = {**row, **{k: json.loads(row[k]) for k in ("task_ref", "request", "model", "code")},
                  "warm_start": warm_start, "schema_version": SCHEMA_VERSION}
        return RunRecorder(self, run_id, header)

    def _update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {"goal", "tolerances", "warm_start", "label", "pinned"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"run fields {sorted(bad)} are not updatable")
        vals = [canonical(v) if k in ("goal", "tolerances", "warm_start") and v is not None else v for k, v in fields.items()]
        with self._lock:
            cur = self._db.execute(f"UPDATE runs SET {','.join(f'{k}=?' for k in fields)} WHERE run_id=?", (*vals, run_id))
            if cur.rowcount != 1:
                raise KeyError(run_id)

    def _insert_state(self, run_id, step, state, metrics, valid, success, reward, score) -> None:
        if not isinstance(state, dict) or not state:
            raise ValueError(f"run {run_id} step {step}: empty state")
        s = canonical(state)
        with self._lock:
            tk = self._db.execute("SELECT task_key FROM runs WHERE run_id=?", (run_id,)).fetchone()["task_key"]
            self._db.execute("INSERT INTO states VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (run_id, int(step), tk, int(bool(valid)), None if success is None else int(bool(success)), s,
                              hashlib.sha256(s.encode()).hexdigest(), canonical(metrics),
                              None if reward is None else float(reward), None if score is None else float(score)))

    def _close_raw(self, run_id: str, live_path: Path, *, status: str, error, success, best_step, n_steps, n_events) -> None:
        if status not in STATUSES or status == "running":
            raise ValueError(f"finish status {status!r}")
        gz = live_path.with_suffix(".jsonl.gz")
        tmp = gz.with_suffix(".gz.tmp")
        with open(live_path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, gz)
        digest, size = sha256_file(gz), gz.stat().st_size
        with self._lock:
            self._db.execute(
                "UPDATE runs SET status=?, error=?, success=?, best_step=?, n_steps=?, n_events=?, raw_sha256=?, raw_bytes=?, "
                "finished_utc=? WHERE run_id=?",
                (status, error, None if success is None else int(bool(success)), best_step, n_steps, n_events, digest, size,
                 utc_now(), run_id))
        live_path.unlink()

    def recover(self, stale_seconds: float = 900.0) -> Dict[str, List[str]]:
        """Close runs a dead server left 'running': gzip what was written, status 'interrupted'.

        Several servers may share one store (e.g. an old and a new demo job), so 'running' alone does not mean dead. A live
        run appends (and flushes) its raw file on every event, so only runs whose raw file has not been written for
        `stale_seconds` are closed; fresher ones are left alone and reported as 'live'. Returns {"closed": [...], "live": [...]}."""
        if not stale_seconds > 0:
            raise ValueError("stale_seconds must be > 0")
        closed, live_ids = [], []
        with self._lock:
            rows = self._db.execute("SELECT run_id FROM runs WHERE status='running' ORDER BY run_id").fetchall()
        for r in rows:
            live = self.root / "raw" / f"{r['run_id']}.jsonl"
            if not live.exists():
                raise RuntimeError(f"run {r['run_id']} is 'running' but {live} is missing")
            if time.time() - live.stat().st_mtime < stale_seconds:
                live_ids.append(r["run_id"])
                continue
            lines = live.read_text(encoding="utf-8").splitlines()
            n_events = sum(1 for ln in lines[1:] if ln.strip())
            n_steps = self._db.execute("SELECT COUNT(*) FROM states WHERE run_id=? AND step>=0", (r["run_id"],)).fetchone()[0]
            self._close_raw(r["run_id"], live, status="interrupted", error="server stopped during the run", success=None,
                            best_step=None, n_steps=n_steps, n_events=n_events)
            closed.append(r["run_id"])
        return {"closed": closed, "live": live_ids}

    def label(self, run_id: str, label: Optional[str] = None, pinned: Optional[bool] = None) -> None:
        f: Dict[str, Any] = {}
        if label is not None:
            if len(label) > 200:
                raise ValueError("label longer than 200 characters")
            f["label"] = label
        if pinned is not None:
            f["pinned"] = int(bool(pinned))
        if f:
            self._update_run(check_run_id(run_id), **f)

    def save(self, run_id: str, label: Optional[str] = None) -> None:
        """Keep a finished run: listed in the database and its states indexed for retrieval."""
        run = self.get_run(check_run_id(run_id))
        if run["status"] == "running":
            raise ValueError(f"run {run_id} is still running; save it when it ends")
        if label is not None:
            self.label(run_id, label=label)
        with self._lock:
            self._db.execute("UPDATE runs SET saved=1, saved_utc=COALESCE(saved_utc, ?) WHERE run_id=?", (utc_now(), run_id))

    def unsave(self, run_id: str) -> None:
        """Take a run out of the database: it is staged again and purged once the TTL has passed since now."""
        self.get_run(check_run_id(run_id))
        with self._lock:
            self._db.execute("UPDATE runs SET saved=0, saved_utc=NULL, finished_utc=? WHERE run_id=?", (utc_now(), run_id))

    def purge_unsaved(self) -> List[str]:
        """Delete unsaved runs finished more than unsaved_ttl_hours ago (none when it is never): state rows, run row, raw file."""
        with self._lock:
            ids = self.purgeable()
            for run_id in ids:
                r = self._db.execute("SELECT raw_path, saved, status FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if r["saved"] or r["status"] == "running":
                    raise RuntimeError(f"purge: {run_id} is saved or running")
                self._db.execute("DELETE FROM states WHERE run_id=?", (run_id,))
                self._db.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
                (self.root / r["raw_path"]).unlink(missing_ok=True)
        return ids

    # ---- reading -------------------------------------------------------------------------------------------------
    @staticmethod
    def _run_dict(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        for k in ("task_ref", "goal", "tolerances", "request", "model", "code", "warm_start"):
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        for k in ("success", "pinned", "saved"):
            if d.get(k) is not None:
                d[k] = bool(d[k])
        return d

    def list_runs(self, task_key: Optional[str] = None, limit: int = 200, saved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Runs, newest first (pinned first), each with best_metrics = the metrics of its best turn and start_state = its
        starting state (None if not recorded)."""
        q = ("SELECT r.*, s.metrics AS best_metrics, s0.state AS start_state FROM runs r "
             "LEFT JOIN states s ON s.run_id = r.run_id AND s.step = r.best_step "
             "LEFT JOIN states s0 ON s0.run_id = r.run_id AND s0.step = -1")
        where, args = [], []
        if task_key is not None:
            where.append("r.task_key=?")
            args.append(task_key)
        if saved is not None:
            where.append("r.saved=?")
            args.append(int(bool(saved)))
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY r.pinned DESC, r.created_utc DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            out = []
            for r in self._db.execute(q, args).fetchall():
                d = self._run_dict(r)
                d["best_metrics"] = json.loads(d["best_metrics"]) if d["best_metrics"] else None
                d["start_state"] = json.loads(d["start_state"]) if d["start_state"] else None
                out.append(d)
            return out

    def get_run(self, run_id: str) -> Dict[str, Any]:
        with self._lock:
            r = self._db.execute("SELECT * FROM runs WHERE run_id=?", (check_run_id(run_id),)).fetchone()
        if r is None:
            raise KeyError(run_id)
        return self._run_dict(r)

    @staticmethod
    def _state_dict(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        d["state"], d["metrics"] = json.loads(d["state"]), json.loads(d["metrics"])
        d["valid"] = bool(d["valid"])
        d["success"] = None if d["success"] is None else bool(d["success"])
        return d

    def get_states(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM states WHERE run_id=? ORDER BY step", (check_run_id(run_id),)).fetchall()
        return [self._state_dict(r) for r in rows]

    def get_state(self, run_id: str, step: int) -> Dict[str, Any]:
        with self._lock:
            r = self._db.execute("SELECT * FROM states WHERE run_id=? AND step=?", (check_run_id(run_id), int(step))).fetchone()
        if r is None:
            raise KeyError(f"{run_id} step {step}")
        return self._state_dict(r)

    def candidate_states(self, task_key: str) -> List[Dict[str, Any]]:
        """Valid states of *saved* runs (each turn's design, plus the initial design where indexable) for a task key, with the
        run's goal and tolerances: the pool the retrieval ranks."""
        with self._lock:
            rows = self._db.execute(
                "SELECT s.*, r.goal AS run_goal, r.tolerances AS run_tolerances, r.created_utc AS run_created, "
                "r.label AS run_label FROM states s JOIN runs r USING (run_id) "
                "WHERE s.task_key=? AND s.valid=1 AND r.saved=1", (task_key,)).fetchall()
        out = []
        for r in rows:
            d = self._state_dict(r)
            d["run_goal"] = json.loads(d["run_goal"]) if d["run_goal"] else None
            d["run_tolerances"] = json.loads(d["run_tolerances"]) if d["run_tolerances"] else None
            out.append(d)
        return out

    def raw_path(self, run_id: str, verify: bool = True) -> Path:
        run = self.get_run(run_id)
        if run["status"] == "running":
            raise RuntimeError(f"run {run_id} is still running")
        p = self.root / run["raw_path"]
        if verify:
            got = sha256_file(p)
            if got != run["raw_sha256"]:
                raise RuntimeError(f"run {run_id}: raw file sha256 {got} != recorded {run['raw_sha256']}")
        return p

    def iter_raw(self, run_id: str) -> Iterator[Tuple[float, Dict[str, Any]]]:
        """(t, event) of a finished run, sha-verified first; the header line is skipped."""
        p = self.raw_path(run_id, verify=True)
        with gzip.open(p, "rt", encoding="utf-8") as f:
            first = json.loads(f.readline())
            if "header" not in first or first["header"]["run_id"] != run_id:
                raise RuntimeError(f"{p}: header does not name run {run_id}")
            for ln in f:
                if ln.strip():
                    o = json.loads(ln)
                    yield float(o["t"]), o["e"]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n_runs = self._db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            n_states = self._db.execute("SELECT COUNT(*) FROM states s JOIN runs r USING (run_id) WHERE s.valid=1 AND r.saved=1").fetchone()[0]
            n_saved = self._db.execute("SELECT COUNT(*) FROM runs WHERE saved=1").fetchone()[0]
            n_ok = self._db.execute("SELECT COUNT(*) FROM runs WHERE saved=1 AND success=1").fetchone()[0]
        return {"root": str(self.root), "domain": self.domain, "saved_runs": n_saved, "saved_successful_runs": n_ok,
                "unsaved_runs": n_runs - n_saved, "indexed_states": n_states, "unsaved_ttl_hours": self.unsaved_ttl_hours}

    def verify_all(self) -> Dict[str, Any]:
        """Every finished run's raw file exists, matches its sha256 and has as many events as recorded."""
        bad = []
        runs = self.list_runs(limit=10 ** 9)
        for r in runs:
            if r["status"] == "running":
                continue
            try:
                n = sum(1 for _ in self.iter_raw(r["run_id"]))
                if n != r["n_events"]:
                    bad.append((r["run_id"], f"{n} events in raw, {r['n_events']} recorded"))
            except Exception as e:   # collected and reported: verify lists every broken run, then the caller fails
                bad.append((r["run_id"], f"{type(e).__name__}: {e}"))
        return {"checked": len(runs), "bad": bad}
