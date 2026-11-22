# Browser UI

```bash
slic data --family standard --family nodes --cards --topologies   # task sets, node cards, schematics
vllm serve <checkpoint> --served-model-name slic-random --port 8000 --limit-mm-per-prompt '{"image":0,"video":0}' &
slic serve --tasks data/standard/test_iid.parquet data/standard/test_ood.parquet --port 8080 \
    --model-url http://127.0.0.1:8000/v1 --model-name slic-random
```

One page (`web_ui/server.py`). The sidebar selects the process node, the circuit (by name, with an NMOS / PMOS input
switch and its schematic), the start design, the goal, the starting sizes and the model. An episode runs the evaluation
pipeline, so the model sees exactly the prompts of `slic eval` (default turn budget 32 on the page, 8 in the protocol).

- Each turn's reply, thinking included, streams into a foldable card; the start design's metrics open the timeline.
- The Parameters tab shows every device parameter from the start through each turn.
- Tolerance (%) sets each spec's own tolerance (gain, power, bandwidth, phase margin), pre-filled with the evaluation
  protocol's; the prompt states them and success and the auto-stop use them (`/api/protocol` gives the defaults).
- Sampling (temperature, top-p, max tokens, seed) opens with the model's defaults. With a seed, a local vLLM model repeats
  an episode exactly; without one, every run samples afresh.
- Record saves a .webm screen capture (browsers allow it only on https or localhost; use an SSH tunnel for a remote server).
- Models: an OpenAI-compatible endpoint (`--model-url`; `LOCAL_MODEL_API_KEY` if it needs a key) or an API provider
  (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...).
- The goal you edit is kept per task across page reloads (in the browser's localStorage), and the page reopens on the
  last task; "reset to task goal" restores the dataset's goal.

The simulator runs in the server process, so start it where ngspice (or Spectre) is installed.

## Experience database (optional)

```bash
slic serve --memory-dir runs_db --tasks data/standard/test_iid.parquet data/standard/test_ood.parquet --port 8080 \
    --model-url http://127.0.0.1:8000/v1 --model-name slic-random
```

With `--memory-dir` the server records every run into that directory (created if missing, on the machine that runs the
server). Without it nothing is recorded and the page is as above. The code is in `web_ui/memory/`, the panel in
`web_ui/static/memory.js`, the Database tab in `web_ui/static/database.html`.

- **Save to database.** Every run is recorded while it runs: the request, every streamed event (tokens included) and each
  turn's design with its metrics. After a run, "Save to database" keeps it. Unsaved runs, including stopped or interrupted
  ones, are deleted a set time after they end: 24 h by default, or never. The Database tab changes this setting; before it
  deletes anything it shows how many runs the new value would remove.
- **Database tab** (`/database`, linked from the header). Saved runs with topology, process and vdd columns, goal, result,
  best turn, model and label, a filter per column, per-turn details, raw download (`.jsonl.gz`, sha256-checked), label / pin,
  remove (back to unsaved), and "use as start" on any turn.
- **Replay.** A recorded run streams back into the page with its original timing or faster (`/?replay=<run_id>`). It needs
  no model and no simulator.
- **Nearest start.** "Find nearest start" ranks the saved designs of the same topology, node and vdd against the current
  goal and tolerances, and fills the chosen one into the starting sizes (`/?warm=<run_id>:<step>` from the Database tab).
  The warm-start banner goes back with "previous start" (the sizes before Use as start) or "task start" (the task's own).
  The distance is the sum over specs of how far a design falls outside its tolerated bound, in decades (gain dB/20; power,
  bandwidth, phase margin log10 of the ratio), so exceeding a minimum costs nothing; "ranking weights" scale each spec.
  A candidate's pass is exact only when its run had the same goal and tolerances; otherwise it is an estimate. The new
  run re-simulates the chosen design and stops with an error if its metrics differ from the stored ones.

Files: `<dir>/memory.sqlite` (runs and the indexed designs) and `<dir>/raw/<run_id>.jsonl.gz` (each run's events). One
server per directory; a server that starts on a directory closes runs a crashed server left open (idle for 15 min).
