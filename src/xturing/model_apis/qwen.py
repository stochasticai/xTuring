from datetime import datetime
from importlib import import_module
from typing import Dict, List, Optional, Sequence

import torch

# `AutoModelForMultimodalLM` was introduced in transformers 5.0.0. On older
# releases the package imports fine but the symbol is missing, which raises
# ImportError rather than ModuleNotFoundError -- catch the broader class so
# importing xturing.model_apis keeps working without Qwen3-Omni support.
try:
    from transformers import AutoModelForMultimodalLM, AutoTokenizer
except ImportError as import_err:  # pragma: no cover - optional dependency
    AutoModelForMultimodalLM = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = import_err
else:  # pragma: no cover - dependency import paths exercised in runtime envs
    _TRANSFORMERS_IMPORT_ERROR = None

from xturing.model_apis.base import TextGenerationAPI


class Qwen3OmniTextGenerationAPI(TextGenerationAPI):
    """Text generation API wrapper for running Qwen3-Omni locally via Hugging Face."""

    config_name = "qwen3_omni"

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
        default_generate_kwargs: Optional[Dict] = None,
    ):
        auto_model_cls, auto_tokenizer_cls = self._ensure_dependency()
        super().__init__(
            engine=model_name_or_path,
            api_key=None,
            request_batch_size=1,
        )
        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}
        self.default_generate_kwargs = default_generate_kwargs or {}

        self.tokenizer = auto_tokenizer_cls.from_pretrained(
            model_name_or_path, trust_remote_code=True, **tokenizer_kwargs
        )
        self.model = auto_model_cls.from_pretrained(
            model_name_or_path, trust_remote_code=True, **model_kwargs
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _ensure_dependency():
        # Resolve from the currently loaded module to stay correct across reloads.
        module = import_module(__name__)
        auto_model_cls = getattr(module, "AutoModelForMultimodalLM", None)
        auto_tokenizer_cls = getattr(module, "AutoTokenizer", None)
        if auto_model_cls is None or auto_tokenizer_cls is None:
            transformers_import_error = getattr(
                module, "_TRANSFORMERS_IMPORT_ERROR", None
            )
            message = (
                "Qwen3OmniTextGenerationAPI requires transformers>=5.0.0, which "
                "provides AutoModelForMultimodalLM. Upgrade with "
                "`pip install 'transformers>=5.0.0'`."
            )
            raise ImportError(message) from transformers_import_error
        return auto_model_cls, auto_tokenizer_cls

    def _trim_stop_sequences(
        self, text: str, stop_sequences: Optional[Sequence[str]]
    ) -> str:
        if not stop_sequences:
            return text
        cut_index = len(text)
        for stop in stop_sequences:
            if not stop:
                continue
            idx = text.find(stop)
            if idx != -1 and idx < cut_index:
                cut_index = idx
        return text[:cut_index].rstrip()

    def _generate_single(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        stop_sequences: Optional[Sequence[str]],
        n: int,
        generation_overrides: Dict,
    ) -> List[Dict[str, str]]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        do_sample = temperature is not None and temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        generate_kwargs.update(self.default_generate_kwargs)
        generate_kwargs.update(generation_overrides)
        outputs = self.model.generate(**inputs, **generate_kwargs)
        if n == 1:
            outputs = outputs.unsqueeze(0) if outputs.dim() == 1 else outputs
        generated_sequences: List[Dict[str, str]] = []
        prompt_length = inputs["input_ids"].shape[-1]
        for sequence in outputs:
            completion_tokens = sequence[prompt_length:]
            text = self.tokenizer.decode(
                completion_tokens,
                skip_special_tokens=True,
            ).strip()
            text = self._trim_stop_sequences(text, stop_sequences)
            generated_sequences.append(
                {
                    "text": text,
                    "finish_reason": "stop",
                }
            )
        return generated_sequences

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
        retries=0,
        **generation_overrides,
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            choices = self._generate_single(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                n=n,
                generation_overrides=generation_overrides,
            )
            data = {
                "prompt": prompt,
                "response": {"choices": choices},
                "created_at": str(datetime.now()),
            }
            results.append(data)

        return results
