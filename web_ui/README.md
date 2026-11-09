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
- Sampling (temperature, top-p, max tokens, seed) opens with the model's defaults. With a seed, a local vLLM model repeats
  an episode exactly; without one, every run samples afresh.
- Record saves a .webm screen capture (browsers allow it only on https or localhost; use an SSH tunnel for a remote server).
- Models: an OpenAI-compatible endpoint (`--model-url`; `LOCAL_MODEL_API_KEY` if it needs a key) or an API provider
  (`OPENAI_API_KEY`, `GEMINI_API_KEY`, ...).

The simulator runs in the server process, so start it where ngspice (or Spectre) is installed.
