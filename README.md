<div align="center">

# SLIC: Reinforcement Fine-Tuning Small LMs for Multi-Turn Analog Circuit Optimization

[![Paper](https://img.shields.io/badge/Paper-arXiv%20(coming%20soon)-b31b1b)](https://arxiv.org/)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Data-slic--agent-yellow)](https://huggingface.co/slic-agent)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

</div>

## 📖 Overview

An LLM agent optimizes an analog circuit over several simulate-and-adjust steps until gain, power, bandwidth and phase
margin meet their targets. This repository contains **CircuitOp**, the multi-step environment behind the released task
sets; **SLIC**, the stepwise RL recipe with online continuation (`docs/METHOD.md`); the baselines, the evaluation protocol,
a `slic` command line and a browser UI.

## 🛠️ Install

```bash
git clone https://github.com/slic-agent/slic-agent.git && cd slic-agent
conda env create -f environment.yml && conda activate slic     # Python 3.12, ngspice, the paper's package versions
pip install --no-deps -r requirements-hopper.txt                # Hopper GPUs (H100 / H200) only
```

The flash-attn / causal-conv1d wheels in `requirements.txt` are built for Linux x86_64 with CUDA 12; on other systems (e.g.
ARM or CUDA 11), replace those two lines with `pip install flash-attn==2.8.1 causal-conv1d==1.7.0 --no-build-isolation`.
To simulate with Cadence Spectre instead of ngspice, see `docs/SIMULATORS.md`.

## 🚀 Quick start

```bash
slic check                                                                   # packages, GPUs, simulator (ngspice / Spectre), one simulation
slic data --family standard                                                  # task sets from the Hub
slic eval  --config configs/eval/standard_iid.yaml --model <checkpoint>      # the paper's IID protocol
slic train --config configs/train/slic_random.yaml --set model_path=<sft>   # the main method (2 GPUs, 400 steps)
```

## 📦 Data

```bash
slic data --family standard --family hard   # data/standard, data/hard (docs/DATASETS.md)
slic data --family nodes --cards            # the 22-130 nm task sets; device model cards -> circuit/dataset/
```

`slic tasks --spec my_tasks.yaml` turns your own circuits into a task parquet in the released format (`docs/DATASETS.md`).

## 🔌 Solve one task

Serve the checkpoint with an OpenAI-compatible server (Qwen3.5 checkpoints need the `--limit-mm-per-prompt` setting) and
stream an episode; it uses the evaluation pipeline, so the model sees exactly the prompts of `slic eval`:

```bash
vllm serve models/my-slic-random --served-model-name slic-random --port 8000 --limit-mm-per-prompt '{"image":0,"video":0}' &
slic solve --config configs/solve/example.yaml --index 3
```

## 📊 Evaluate

```bash
slic eval --config configs/eval/standard_ood.yaml --model models/my-checkpoint
```

`configs/eval/` has one file per task set (8 steps, 3 samples per task, Pass@1 / Pass@3). Results go to
`outputs/eval/<run_tag>/results.json` with per-topology breakdowns and the raw rollouts; one 80 GB GPU suffices.

## 🏋️ Train

```bash
slic train --config configs/train/slic_random.yaml              # main method
slic train --config configs/train/mt_grpo.yaml                  # MT-GRPO baseline
slic train --config configs/train/slic_random.yaml --dry-run    # print the resolved command
```

`configs/train/` has one file per configuration in the paper (`docs/REPRODUCE.md` maps them to the tables); each lists
only its overrides of `scripts/train/run_slic.sh`, and `--set key=value` adds one. Checkpoints and rollout dumps go to `outputs/`;
`wandb: true` logs to Weights & Biases. RL starts from an SFT checkpoint: `slic train --config configs/sft/qwen3.5_4b.yaml`
(`scripts/train/sft/README.md`).

## 🖥️ Browser UI

```bash
slic data --family nodes --cards --topologies          # optional: node switch and schematics
slic serve --tasks data/standard/test_iid.parquet data/standard/test_ood.parquet --port 8080 \
    --model-url http://127.0.0.1:8000/v1 --model-name slic-random
```

`--model-url` is the model server started above (`vllm serve --port 8000`); `--port` is the page (default 8080), opened at
`http://<host>:8080` (`web_ui/README.md`).

## 🗂️ Layout

```
slic/            command line (check / solve / eval / train / data / tasks / serve)
configs/         paper configurations: train/, eval/, sft/, solve/
scripts/         RL / SFT and evaluation launchers (train/, eval/), checkpoint export (checkpoint/)
agent_system/    environment manager, multi-step rollout with online continuation, the CircuitOp environment
verl/            trainer (SLIC advantage estimator in verl/trainer/ppo/)
circuit/         netlist generation, simulator runners, task builders, device model cards (dataset/), API-model agent
web_ui/          browser UI
docs/            METHOD.md, DATASETS.md, REPRODUCE.md, SIMULATORS.md
```

## 📝 Citation

The BibTeX entry follows with the arXiv release.

## 🙏 Acknowledgement

We implement our reinforcement learning recipe extending from [verl](https://github.com/volcengine/verl) and
[verl-agent](https://github.com/langfengQ/verl-agent), with [vLLM](https://github.com/vllm-project/vllm) for rollouts and
[ngspice](https://ngspice.sourceforge.io/) and Cadence Spectre for circuit simulation; the 22–130 nm device models are the
[Predictive Technology Model](https://ptm.asu.edu/) cards. We thank the authors of these projects.
