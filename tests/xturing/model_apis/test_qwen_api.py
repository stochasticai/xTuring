from types import SimpleNamespace

import torch


class DummyTokenizer:
    def __init__(self, decoded_text="Generated response."):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.decoded_text = decoded_text

    def __call__(self, text, return_tensors=None):
        input_ids = torch.tensor([[11, 12]])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, tokens, skip_special_tokens=True):
        return self.decoded_text


class DummyModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.last_kwargs = None

    def to(self, device):
        self.device = device
        return self

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.last_kwargs = kwargs
        prompt_len = input_ids.shape[-1] if input_ids is not None else 0
        total_len = prompt_len + 2
        base = torch.arange(total_len).unsqueeze(0).long()
        num_sequences = kwargs.get("num_return_sequences", 1)
        return base.repeat(num_sequences, 1)


def _install_mocks(monkeypatch, tokenizer):
    dummy_tokenizer = tokenizer
    dummy_model = DummyModel()
    monkeypatch.setattr(
        "xturing.model_apis.qwen.AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_, **__: dummy_tokenizer),
        raising=False,
    )
    monkeypatch.setattr(
        "xturing.model_apis.qwen.AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_, **__: dummy_model),
        raising=False,
    )
    return dummy_tokenizer, dummy_model


def test_qwen3_omni_initialization(monkeypatch):
    from xturing.model_apis.qwen import Qwen3OmniTextGenerationAPI

    tokenizer, model = _install_mocks(monkeypatch, DummyTokenizer())
    api = Qwen3OmniTextGenerationAPI(model_name_or_path="local-qwen", device="cpu")

    assert api.engine == "local-qwen"
    assert api.tokenizer is tokenizer
    assert api.model is model
    assert str(api.device) == "cpu"


def test_qwen3_omni_generate_text(monkeypatch):
    from xturing.model_apis.qwen import Qwen3OmniTextGenerationAPI

    tokenizer, model = _install_mocks(monkeypatch, DummyTokenizer("Hello world."))

    api = Qwen3OmniTextGenerationAPI(model_name_or_path="local-qwen", device="cpu")
    results = api.generate_text(
        prompts="Hi",
        max_tokens=16,
        temperature=0.7,
        top_p=0.9,
        n=2,
    )

    assert len(results) == 1
    response = results[0]["response"]
    assert len(response["choices"]) == 2
    for choice in response["choices"]:
        assert choice["text"] == "Hello world."
        assert choice["finish_reason"] == "stop"

    assert model.last_kwargs["max_new_tokens"] == 16
    assert model.last_kwargs["temperature"] == 0.7
    assert model.last_kwargs["top_p"] == 0.9
    assert model.last_kwargs["num_return_sequences"] == 2


def test_qwen3_omni_stop_sequences(monkeypatch):
    from xturing.model_apis.qwen import Qwen3OmniTextGenerationAPI

    tokenizer, _ = _install_mocks(monkeypatch, DummyTokenizer("Answer: hello<eot>"))

    api = Qwen3OmniTextGenerationAPI(model_name_or_path="local-qwen", device="cpu")
    results = api.generate_text(
        prompts="Question?",
        max_tokens=8,
        temperature=0.0,
        stop_sequences=["<eot>"],
        n=1,
    )

    assert results[0]["response"]["choices"][0]["text"] == "Answer: hello"


def test_qwen3_omni_registered():
    from xturing.model_apis import BaseApi
    from xturing.model_apis.qwen import Qwen3OmniTextGenerationAPI

    assert (
        BaseApi.registry[Qwen3OmniTextGenerationAPI.config_name]
        is Qwen3OmniTextGenerationAPI
    )
