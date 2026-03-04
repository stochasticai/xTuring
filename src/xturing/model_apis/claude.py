import time
from datetime import datetime
from importlib import import_module

try:
    from anthropic import Anthropic
    from anthropic import APIConnectionError as AnthropicAPIConnectionError
    from anthropic import APIError as AnthropicAPIError
    from anthropic import RateLimitError as AnthropicRateLimitError
except ModuleNotFoundError as import_err:  # pragma: no cover - optional dependency
    Anthropic = None
    AnthropicAPIError = AnthropicAPIConnectionError = AnthropicRateLimitError = (
        Exception
    )
    _ANTHROPIC_IMPORT_ERROR = import_err
else:  # pragma: no cover - dependency import paths exercised in runtime envs
    _ANTHROPIC_IMPORT_ERROR = None

from xturing.model_apis.base import TextGenerationAPI


class ClaudeTextGenerationAPI(TextGenerationAPI):
    config_name = "claude"

    def __init__(self, model, api_key, request_batch_size=1):
        anthropic_client_cls = self._ensure_dependency()
        super().__init__(
            engine=model, api_key=api_key, request_batch_size=request_batch_size
        )
        self._client = anthropic_client_cls(api_key=api_key)

    @staticmethod
    def _ensure_dependency():
        # Resolve from the currently loaded module to stay correct across reloads.
        module = import_module(__name__)
        anthropic_client_cls = getattr(module, "Anthropic", None)
        if anthropic_client_cls is None:
            anthropic_import_error = getattr(module, "_ANTHROPIC_IMPORT_ERROR", None)
            message = (
                "The anthropic SDK is required for ClaudeTextGenerationAPI. "
                "Install it with `pip install anthropic`."
            )
            raise ModuleNotFoundError(message) from anthropic_import_error
        return anthropic_client_cls

    def _make_request(self, prompt, max_tokens, temperature, top_p, stop_sequences):
        params = {
            "model": self.engine,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if top_p is not None:
            params["top_p"] = top_p
        if stop_sequences:
            params["stop_sequences"] = stop_sequences
        return self._client.messages.create(**params)

    @staticmethod
    def _render_response(response):
        if response is None:
            return None
        text_chunks = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                text_chunks.append(getattr(block, "text", ""))
        predicts = {
            "choices": [
                {
                    "text": "".join(text_chunks),
                    "finish_reason": getattr(response, "stop_reason", "eos"),
                }
            ]
        }
        return predicts

    def generate_text(
        self,
        prompts,
        max_tokens,
        temperature,
        top_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        stop_sequences=None,
        logprobs=None,
        n=1,
        best_of=1,
        retries=3,
        **kwargs,
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            response = None
            retry_cnt = 0
            backoff_time = 30
            while retry_cnt <= retries:
                try:
                    response = self._make_request(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_sequences=stop_sequences,
                    )
                    break
                except (
                    AnthropicAPIError,
                    AnthropicAPIConnectionError,
                    AnthropicRateLimitError,
                ) as e:
                    print(f"ClaudeError: {e}.")
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                    retry_cnt += 1

            data = {
                "prompt": prompt,
                "response": self._render_response(response),
                "created_at": str(datetime.now()),
            }
            results.append(data)

        return results


class ClaudeSonnet(ClaudeTextGenerationAPI):
    config_name = "claude_3_sonnet"

    def __init__(self, api_key, request_batch_size=1):
        super().__init__(
            model="claude-3-sonnet-20240229",
            api_key=api_key,
            request_batch_size=request_batch_size,
        )
