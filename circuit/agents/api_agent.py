"""API-model agent for the browser UI and `slic solve` (`model: {provider: openai | gemini | claude | grok | deepseek |
openrouter | xiaomi}`): one chat request per turn with the system message and observation the environment loop builds;
`provider: local` (any OpenAI-compatible server, e.g. `vllm serve <checkpoint>`) does not use this module."""

import json
import os
import urllib.error
import urllib.request

try:  # optional: only required for provider="gemini" (pip install google-genai)
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = types = None

from agent_system.environments.prompts.circuitop import CIRCUITOP_SYSTEM_HEADER as SYSTEM_HEADER   # the training header


class Agent:
    def __init__(
        self,
        provider="openai",
        model_name=None,
        max_output_tokens=None,
        deepseek_thinking_mode="enabled",
        deepseek_reasoning_effort="high",
        gemini_thinking_level="high",
        sampling=None,
    ):
        """
        Args:
            provider: openai, gemini, claude, grok, deepseek, openrouter or xiaomi (API key from the provider's usual
                environment variable: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, DEEPSEEK_API_KEY,
                OPENROUTER_API_KEY, XIAOMI_MIMO_API_KEY)
            model_name: the provider's model id (default: the provider's entry below)
            max_output_tokens: output cap passed to the API (default: the provider's own)
            deepseek_thinking_mode / deepseek_reasoning_effort: thinking switch and effort for DeepSeek and OpenRouter
            gemini_thinking_level: Gemini thinking level (None or "off" disables thinking)
            sampling: optional {temperature, top_p, max_tokens, seed}; a key left out keeps this agent's default
                (temperature 0.4, the provider's top-p, max_output_tokens, no seed). Claude has no seed: setting one fails.
        """
        self.sampling = {k: v for k, v in (sampling or {}).items() if v is not None}
        unknown = sorted(set(self.sampling) - {"temperature", "top_p", "max_tokens", "seed"})
        if unknown:
            raise ValueError(f"unknown sampling keys {unknown}")
        self.provider = provider.lower()
        self.max_output_tokens = max_output_tokens
        self.system_header_text = SYSTEM_HEADER
        self.deepseek_thinking_mode = (deepseek_thinking_mode or "enabled").lower()
        self.deepseek_reasoning_effort = (deepseek_reasoning_effort or "high").lower()
        self.gemini_thinking_level = (
            None if gemini_thinking_level is None else gemini_thinking_level.upper()
        )
        
        if self.provider == "openai":
            from openai import OpenAI
            self.model_name = model_name or "gpt-4o"
            self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            
        elif self.provider == "gemini":
            if genai is None:
                raise ImportError('provider="gemini" requires the google-genai package (pip install google-genai)')
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model_name = model_name or 'gemini-3.1-pro-preview'
            
        elif self.provider == "claude":
            self.model_name = model_name or "claude-opus-4-6"
            self.api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")

        elif self.provider == "grok":
            from openai import OpenAI
            self.model_name = model_name or "grok-4-1-fast-reasoning"
            self.client = OpenAI(
                api_key=os.environ.get('XAI_API_KEY'),
                base_url="https://api.x.ai/v1"
            )
        
        # deepseek-reasoner (thinking mode)
        # deepseek-chat (non-thinking mode)
        elif self.provider == "deepseek":
            from openai import OpenAI
            self.model_name = model_name or "deepseek-chat"
            self.client = OpenAI(
                api_key=os.environ.get('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com"
            )
            
        elif self.provider == "openrouter":
            from openai import OpenAI
            # Default to FREE Xiaomi MiMo-V2-Flash model
            self.model_name = model_name or "xiaomi/mimo-v2-flash:free"
            self.client = OpenAI(
                api_key=os.environ.get('OPENROUTER_API_KEY'),
                base_url="https://openrouter.ai/api/v1"
            )
            
        elif self.provider == "xiaomi":
            from openai import OpenAI
            # Direct Xiaomi MiMo API (FREE at platform.xiaomimimo.com)
            self.model_name = model_name or "mimo-v2-flash"
            self.client = OpenAI(
                api_key=os.environ.get('XIAOMI_MIMO_API_KEY'),
                base_url="https://api.xiaomimimo.com/v1"
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Choose from: "
                "openai, gemini, claude, grok, deepseek, openrouter, xiaomi"
            )
        
    def get_action(self, obs, system_message=None):
        """Get action from LLM based on observation.
        
        Args:
            obs: User observation text
            system_message: Optional system message (circuit structure, etc.)
            
        Returns:
            (raw_output, display_text, token_info): the response text the environment receives, the same text with any
            separate reasoning shown first, and the request's token / character counts.
        """
        if self.provider == "gemini":
            return self._get_action_gemini(obs, system_message)
        if self.provider == "claude":
            return self._get_action_claude(obs, system_message)
        else:
            return self._get_action_openai_compatible(obs, system_message)
    
    def _get_action_openai_compatible(self, obs, system_message=None):
        """OpenAI-compatible chat API (OpenAI, Grok, DeepSeek, OpenRouter, Xiaomi)."""
        messages = []
        
        # Add system message if provided (circuit structure)
        if system_message:
            messages.append({
                "role": "system",
                "content": self.system_header_text + system_message
            })
        
        # Add user observation
        messages.append({
            "role": "user", 
            "content": obs
        })
        
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(self.sampling.get("temperature", 0.4)),
            "n": 1,
            "stop": None,
        }
        if "top_p" in self.sampling:
            request_kwargs["top_p"] = float(self.sampling["top_p"])
        if "seed" in self.sampling:   # OpenAI-compatible seed: reproducible on vLLM, best effort on hosted APIs
            request_kwargs["seed"] = int(self.sampling["seed"])
        # honored by all OpenAI-compatible providers; on OpenRouter an unset
        # max_tokens also inflates the per-request credit pre-authorization
        max_tokens = self.sampling.get("max_tokens", self.max_output_tokens)
        if max_tokens is not None:
            request_kwargs["max_tokens"] = int(max_tokens)
        if self.provider == "deepseek":
            request_kwargs["extra_body"] = {
                "thinking": {"type": self.deepseek_thinking_mode}
            }
            if self.deepseek_thinking_mode == "enabled":
                request_kwargs["reasoning_effort"] = self.deepseek_reasoning_effort
        elif self.provider == "openrouter":
            # request thinking via extra_body.reasoning; deepseek_thinking_mode /
            # deepseek_reasoning_effort double as the provider-agnostic knobs.
            extra_body = {}
            if self.deepseek_thinking_mode == "enabled":
                extra_body["reasoning"] = {"effort": self.deepseek_reasoning_effort}
            # OpenRouter hosts differ in quantization; OPENROUTER_PROVIDER_PREFS is a JSON routing preference,
            # e.g. {"quantizations":["fp8","bf16"]}.
            prefs = os.environ.get("OPENROUTER_PROVIDER_PREFS")
            if prefs:
                extra_body["provider"] = json.loads(prefs)
            if extra_body:
                request_kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request_kwargs)

        # Raw output (logged and sent to the env). content is None when generation is truncated inside the reasoning
        # phase; return an empty (invalid) action instead of crashing the batch.
        raw_output = (response.choices[0].message.content or "").strip()

        # Reasoning/thinking content: DeepSeek returns `reasoning_content`,
        # OpenRouter normalizes most reasoning models to `reasoning`.
        reasoning_content = (getattr(response.choices[0].message, 'reasoning_content', None)
                             or getattr(response.choices[0].message, 'reasoning', None))
        
        if reasoning_content:
            # Display with separated reasoning for terminal
            display_text = f"[REASONING]\n{reasoning_content}\n\n[ANSWER]\n{raw_output}"
        else:
            # No separate reasoning field, display same as raw
            display_text = raw_output

        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
        token_info = {
            "api_model": getattr(response, "model", None),
            "api_completion_id": getattr(response, "id", None),
            "prompt_chars": sum(len(m.get("content", "")) for m in messages),
            "response_chars": len(raw_output),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "response_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "reasoning_content": reasoning_content,
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None)
            if completion_details else None,
            # OpenRouter: the host that actually served the request (hosts differ in quantization)
            "api_provider": getattr(response, "provider", None)
            or (response.model_extra.get("provider")
                if getattr(response, "model_extra", None) else None),
        }
        return raw_output, display_text, token_info

    def _get_action_claude(self, obs, system_message=None):
        """Get action from Anthropic Claude Messages API."""
        if not self.api_key:
            raise ValueError(
                "Claude provider requires ANTHROPIC_API_KEY or CLAUDE_API_KEY to be set."
            )

        if "seed" in self.sampling:
            raise ValueError("the Claude Messages API has no seed; leave the seed empty for provider claude")
        payload = {
            "model": self.model_name,
            "max_tokens": int(self.sampling.get("max_tokens", 4096)),
            "temperature": float(self.sampling.get("temperature", 0.4)),
            "messages": [
                {
                    "role": "user",
                    "content": obs,
                }
            ],
        }
        if "top_p" in self.sampling:
            payload["top_p"] = float(self.sampling["top_p"])
        if system_message:
            payload["system"] = self.system_header_text + system_message

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Claude API request failed ({exc.code}): {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Claude API connection failed: {exc.reason}") from exc

        thinking_parts = []
        answer_parts = []
        for block in response_data.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking" and block.get("thinking"):
                thinking_parts.append(block["thinking"])
            elif block.get("type") == "text" and block.get("text"):
                answer_parts.append(block["text"])

        raw_output = "".join(answer_parts).strip()
        if not raw_output:
            raise RuntimeError(f"Claude response did not contain text output: {response_data}")

        thinking_text = "".join(thinking_parts).strip()
        if thinking_text:
            display_text = f"[THINKING]\n{thinking_text}\n\n[ANSWER]\n{raw_output}"
        else:
            display_text = raw_output

        usage = response_data.get("usage", {})
        token_info = {
            "prompt_chars": len(obs) + (len(self.system_header_text) + len(system_message) if system_message else 0),
            "response_chars": len(raw_output),
            "prompt_tokens": usage.get("input_tokens"),
            "response_tokens": usage.get("output_tokens"),
        }
        return raw_output, display_text, token_info

    def _get_action_gemini(self, obs, system_message=None):
        """Gemini API (google-genai SDK)."""
        if system_message:
            full_prompt = f"System Instructions:\n{self.system_header_text + system_message}\n\nUser Request:\n{obs}"
        else:
            full_prompt = obs
        
        # Build the request as google-genai types.Content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=full_prompt),
                ],
            ),
        ]
        
        # Configure generation without googleSearch tool; thinking can be toggled.
        config_kwargs = {}
        if self.gemini_thinking_level not in (None, "OFF"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.gemini_thinking_level,
            )
        max_tokens = self.sampling.get("max_tokens", self.max_output_tokens)
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_tokens)
        for key, name in (("temperature", "temperature"), ("top_p", "top_p"), ("seed", "seed")):
            if key in self.sampling:
                config_kwargs[name] = self.sampling[key]
        generate_content_config = types.GenerateContentConfig(**config_kwargs)
        
        # Stream response and collect output (including thinking if available)
        thinking_parts = []
        answer_parts = []
        usage_meta = None

        for chunk in self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        ):
            # Gemini thinking mode may expose thinking in separate field
            if hasattr(chunk, 'thinking') and chunk.thinking:
                thinking_parts.append(chunk.thinking)
            if chunk.text:
                answer_parts.append(chunk.text)
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                usage_meta = chunk.usage_metadata

        # Raw output is what gets logged and sent to environment
        raw_output = ''.join(answer_parts).strip()
        thinking_text = ''.join(thinking_parts).strip()

        # Display text separates thinking for terminal viewing
        if thinking_text:
            display_text = f"[THINKING]\n{thinking_text}\n\n[ANSWER]\n{raw_output}"
        else:
            display_text = raw_output

        token_info = {
            "prompt_chars": len(full_prompt),
            "response_chars": len(raw_output),
            "prompt_tokens": getattr(usage_meta, "prompt_token_count", None) if usage_meta else None,
            "response_tokens": getattr(usage_meta, "candidates_token_count", None) if usage_meta else None,
            "thinking_tokens": getattr(usage_meta, "thoughts_token_count", None) if usage_meta else None,
            "total_tokens": getattr(usage_meta, "total_token_count", None) if usage_meta else None,
            "reasoning_content": thinking_text or None,
        }
        return raw_output, display_text, token_info
