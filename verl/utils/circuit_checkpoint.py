"""Non-optional checkpoint retention for CircuitOp RL training."""


CIRCUIT_CHECKPOINT_POLICY = {
    "save_freq": 50,
    "save_best_val_checkpoint": True,
    "max_actor_ckpt_to_keep": 1,
    "max_best_ckpt_to_keep": 1,
}
# Four-spec success, never two-spec `won`. `val/success_rate` scores the final state; presets that continue after
# success (contopt) select by the any-turn `val/performance_success_rate_any_step` (equal when episodes stop at success).
CIRCUIT_VAL_SCORE_KEYS = ("val/success_rate", "val/performance_success_rate_any_step")


def validate_circuit_checkpoint_policy(config):
    if "circuit" not in str(config.env.env_name).lower() or config.trainer.get("val_only", False):
        return
    for key, expected in CIRCUIT_CHECKPOINT_POLICY.items():
        if config.trainer.get(key) != expected:
            raise ValueError(f"CircuitOp checkpoint policy is locked: trainer.{key} must be {expected!r}")
    if config.trainer.get("val_score_key") not in CIRCUIT_VAL_SCORE_KEYS:
        raise ValueError(f"CircuitOp checkpoint policy is locked: trainer.val_score_key must be one of {CIRCUIT_VAL_SCORE_KEYS}")
    if not config.trainer.get("rollout_data_dir") or not config.trainer.get("validation_data_dir"):
        raise ValueError("CircuitOp training requires dedicated raw train and validation dumps")
    if int(config.trainer.get("test_freq", 0)) <= 0:
        raise ValueError("CircuitOp best-checkpoint selection requires periodic validation")
    if config.trainer.get("remove_previous_ckpt_in_save", False):
        raise ValueError("CircuitOp cannot override the one-latest checkpoint retention")
