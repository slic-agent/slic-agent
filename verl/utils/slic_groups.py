"""Merge the public SLIC configuration groups into the single group the trainer reads.

The public config splits the settings by function:

* ``algorithm.slic``      -- online continuation, i.e. the SLIC algorithm itself (``adv_estimator: slic``)
* ``algorithm.step_wise`` -- the per-turn reward, the turn-level advantage shaping and the training-batch assembly

The trainer and the rollout loop read one flat group, ``algorithm.slic``. ``merge_slic_groups`` folds ``step_wise`` into it
before the trainer is built. A key may live in only one group; a duplicate is a configuration error.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf, open_dict

MERGED_GROUPS = ("step_wise",)


def merge_slic_groups(config: DictConfig) -> DictConfig:
    alg = config.algorithm
    base = alg.get("slic")
    if base is None:
        raise ValueError("config.algorithm.slic is missing; the public ppo_trainer.yaml defines it.")
    seen = {k: "slic" for k in base.keys()}
    parts = []
    for name in MERGED_GROUPS:
        part = alg.get(name)
        if part is None:
            continue
        for k in part.keys():
            if k in seen:
                raise ValueError(f"config key '{k}' is set in both algorithm.{seen[k]} and algorithm.{name}; keep it in one group.")
            seen[k] = name
        parts.append(part)
    with open_dict(alg):
        alg.slic = OmegaConf.merge(base, *parts)
    return config
