from fastapi.testclient import TestClient

from xturing.cli import api as api_module


class _DummyGenerationConfig:
    def __init__(self):
        self.penalty_alpha = None
        self.top_k = None
        self.top_p = None
        self.do_sample = None
        self.max_new_tokens = None


class _DummyModel:
    model_name = "dummy-model"

    def __init__(self):
        self._config = _DummyGenerationConfig()
        self.last_texts = []

    def generation_config(self):
        return self._config

    def generate(self, texts):
        self.last_texts = texts
        return [f"echo:{text}" for text in texts]


def _client_with_model():
    api_module.model = _DummyModel()
    return TestClient(api_module.app), api_module.model


def test_legacy_api_accepts_prompt_list_without_nesting():
    client, loaded_model = _client_with_model()

    response = client.post(
        "/api",
        json={
            "prompt": ["hello", "world"],
            "params": {
                "top_p": 0.8,
                "max_new_tokens": 16,
                "do_sample": True,
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["response"] == ["echo:hello", "echo:world"]
    assert loaded_model.last_texts == ["hello", "world"]


def test_openai_models_endpoint_returns_loaded_model():
    client, _ = _client_with_model()

    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "dummy-model"


def test_openai_chat_completions_returns_compatible_shape():
    client, loaded_model = _client_with_model()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy-model",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hi"},
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "dummy-model"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"].startswith("echo:")
    assert loaded_model.last_texts == ["system: Be concise.\nuser: Say hi"]


def test_openai_chat_completions_rejects_empty_messages():
    client, _ = _client_with_model()

    response = client.post(
        "/v1/chat/completions",
        json={"messages": []},
    )

    assert response.status_code == 400
    assert "messages must not be empty" in response.text
