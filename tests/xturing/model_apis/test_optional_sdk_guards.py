"""The optional SDK wrappers must import without their SDK installed.

`xturing.model_apis` is imported transitively by `xturing.datasets`, so a bare
top-level `import cohere` / `import openai` makes the whole package
unimportable in environments where those SDKs are absent.
"""

import pytest


def test_cohere_module_imports_without_sdk():
    module = pytest.importorskip("xturing.model_apis.cohere")
    # Bound at import time, so the retry loop's except clause stays valid.
    assert hasattr(module, "CohereError")
    assert issubclass(module.CohereError, BaseException)


def test_openai_module_imports_without_sdk():
    module = pytest.importorskip("xturing.model_apis.openai")
    assert hasattr(module, "OpenAIError")
    assert issubclass(module.OpenAIError, BaseException)


def test_cohere_api_reports_missing_sdk(monkeypatch):
    from xturing.model_apis.cohere import CohereTextGenerationAPI

    monkeypatch.setattr("xturing.model_apis.cohere.cohere", None)
    with pytest.raises(ModuleNotFoundError, match="pip install cohere"):
        CohereTextGenerationAPI("medium", "fake-key")


def test_openai_api_reports_missing_sdk(monkeypatch):
    from xturing.model_apis.openai import OpenAITextGenerationAPI

    monkeypatch.setattr("xturing.model_apis.openai.openai", None)
    with pytest.raises(ModuleNotFoundError, match="pip install openai"):
        OpenAITextGenerationAPI("davinci", "fake-key", None)
