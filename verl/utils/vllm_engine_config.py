"""Pure validation for vLLM engine construction."""


def validate_prefill_budget(max_num_batched_tokens, max_model_len, enable_chunked_prefill):
    if max_num_batched_tokens <= 0 or max_model_len <= 0:
        raise ValueError('vLLM token budget and model context length must be positive')
    if max_num_batched_tokens < max_model_len and not enable_chunked_prefill:
        raise ValueError(
            'Without chunked prefill, max_num_batched_tokens must cover max_model_len; '
            'increase the token budget or enable chunked prefill'
        )
