"""Fail before optimization when rollout and actor token likelihoods disagree."""

import math

import torch


def validate_rollout_logprobs(rollout, actor, mask, mean_limit, p99_limit):
    if rollout.shape != actor.shape or actor.shape != mask.shape:
        raise ValueError('Rollout/actor logprob and mask shapes must match')
    if any(not math.isfinite(value) or value <= 0 for value in (mean_limit, p99_limit)):
        raise ValueError('Logprob guard thresholds must be finite and positive')
    selected = mask.bool()
    if not selected.any():
        raise ValueError('Logprob guard requires nonempty response tokens')
    rollout = rollout[selected].float()
    actor = actor[selected].float()
    if not torch.isfinite(rollout).all() or not torch.isfinite(actor).all():
        raise FloatingPointError('Nonfinite actor or rollout logprob')
    differences = (actor - rollout).abs()
    mean = differences.mean().item()
    p99 = torch.quantile(differences, .99).item()
    if mean > mean_limit or p99 > p99_limit:
        raise RuntimeError(f'Rollout/actor logprob mismatch before optimizer update: mean={mean}, p99={p99}; limits={mean_limit}/{p99_limit}')
    return {'training/logprob_abs_diff_mean': mean, 'training/logprob_abs_diff_p99': p99,
            'training/logprob_abs_diff_max': differences.max().item()}
