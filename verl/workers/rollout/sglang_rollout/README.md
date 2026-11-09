# SGLang rollout backend — evaluation only

This directory is verl's SGLang rollout, kept as an alternative inference engine for **evaluation**
(`rollout.name=sglang` in a `configs/eval/*.yaml` `overrides:` list). It is not used for any reported result; the
reported numbers come from the vLLM engine.

It needs its own environment: `sglang==0.4.6.post5` pins `torch==2.6`, so install `requirements-sglang.txt` first and
then the package with the frozen constraints, in a conda environment of its own, never in the vLLM one:
`conda create -n slic-sglang -c conda-forge python=3.12 pip ngspice=41 && conda activate slic-sglang &&
pip install -r requirements-sglang.txt && pip freeze > c.txt && pip install -e . -c c.txt`.

Known limits (from smoke tests):

- **Evaluation works for Qwen3-class checkpoints** and reproduces the vLLM numbers (Qwen3-8B, 16 IID tasks, n = 1:
  success 0.1875 with both engines).
- **Qwen3.5 checkpoints cannot be loaded**: sglang 0.4.6 pins `transformers` 4.51, which predates the `qwen3_5` model type.
- **RL training with SGLang is not supported.** sglang 0.4.6 returns a log-probability of `-FLT_MAX` for about 0.05 % of the
  sampled tokens (non-EOS tokens anywhere in the response); the trainer's rollout-log-probability guard
  (`algorithm.rollout_logprob_guard`, on in every released run) therefore rejects every step. Train with vLLM.
