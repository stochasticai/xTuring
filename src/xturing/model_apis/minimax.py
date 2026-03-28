import time
from datetime import datetime

try:
    from openai import OpenAI
    from openai import APIConnectionError as OpenAIAPIConnectionError
    from openai import APIError as OpenAIAPIError
    from openai import RateLimitError as OpenAIRateLimitError
except ModuleNotFoundError as import_err:  # pragma: no cover - optional dependency
    OpenAI = None
    OpenAIAPIError = OpenAIAPIConnectionError = OpenAIRateLimitError = Exception
    _OPENAI_IMPORT_ERROR = import_err
else:  # pragma: no cover - dependency import paths exercised in runtime envs
    _OPENAI_IMPORT_ERROR = None

from xturing.model_apis.base import TextGenerationAPI

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxTextGenerationAPI(TextGenerationAPI):
    config_name = "minimax"

    def __init__(self, model, api_key, request_batch_size=1):
        openai_cls = self._ensure_dependency()
        super().__init__(
            engine=model, api_key=api_key, request_batch_size=request_batch_size
        )
        self._client = openai_cls(
            api_key=api_key, base_url=_MINIMAX_BASE_URL
        )

    @staticmethod
    def _ensure_dependency():
        import importlib

        module = importlib.import_module(__name__)
        openai_cls = getattr(module, "OpenAI", None)
        if openai_cls is None:
            openai_import_error = getattr(module, "_OPENAI_IMPORT_ERROR", None)
            message = (
                "The openai SDK is required for MiniMaxTextGenerationAPI. "
                "Install it with `pip install openai`."
            )
            raise ModuleNotFoundError(message) from openai_import_error
        return openai_cls

    def _clamp_temperature(self, temperature):
        if temperature is not None and temperature <= 0.0:
            return 0.01
        return temperature

    def _make_request(self, prompt, max_tokens, temperature, top_p, stop_sequences):
        params = {
            "model": self.engine,
            "max_tokens": max_tokens,
            "temperature": self._clamp_temperature(temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        if top_p is not None:
            params["top_p"] = top_p
        if stop_sequences:
            params["stop"] = stop_sequences
        return self._client.chat.completions.create(**params)

    @staticmethod
    def _render_response(response):
        if response is None:
            return None
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return None
        text = choice.message.content or ""
        predicts = {
            "choices": [
                {
                    "text": text,
                    "finish_reason": choice.finish_reason or "stop",
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
                    OpenAIAPIError,
                    OpenAIAPIConnectionError,
                    OpenAIRateLimitError,
                ) as e:
                    print(f"MiniMaxError: {e}.")
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


class MiniMaxM27(MiniMaxTextGenerationAPI):
    config_name = "minimax_m2_7"

    def __init__(self, api_key, request_batch_size=1):
        super().__init__(
            model="MiniMax-M2.7",
            api_key=api_key,
            request_batch_size=request_batch_size,
        )


class MiniMaxM27HighSpeed(MiniMaxTextGenerationAPI):
    config_name = "minimax_m2_7_highspeed"

    def __init__(self, api_key, request_batch_size=1):
        super().__init__(
            model="MiniMax-M2.7-highspeed",
            api_key=api_key,
            request_batch_size=request_batch_size,
        )
