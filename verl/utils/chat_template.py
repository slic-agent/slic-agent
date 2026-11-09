# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _token_ids(rendered) -> list[int]:
    # transformers >=5 returns a BatchEncoding (4.x a list); the length-diff arithmetic
    # below needs the plain id list, or the system-prompt strip silently becomes a no-op.
    if hasattr(rendered, "input_ids"):
        return list(rendered.input_ids)
    if isinstance(rendered, dict):
        return list(rendered["input_ids"])
    return rendered


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    token1, token2 = _token_ids(token1), _token_ids(token2)
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    token3 = _token_ids(token3)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt