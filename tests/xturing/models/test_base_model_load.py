import pytest

from xturing.models import BaseModel


class _DummyModel:
    def __init__(self, weights_path=None, model_name=None, **kwargs):
        self.weights_path = weights_path
        self.model_name = model_name
        self.kwargs = kwargs


def test_load_local_dir_without_xturing_config_with_model_name(tmp_path, monkeypatch):
    local_weights = tmp_path / "hf-local-model"
    local_weights.mkdir()

    monkeypatch.setitem(BaseModel.registry, "dummy_model", _DummyModel)

    loaded = BaseModel.load(str(local_weights), model_name="dummy_model", revision="main")

    assert isinstance(loaded, _DummyModel)
    assert loaded.weights_path == local_weights
    assert loaded.model_name is None
    assert loaded.kwargs["revision"] == "main"


def test_load_local_dir_without_xturing_config_requires_model_name(tmp_path):
    local_weights = tmp_path / "hf-local-model"
    local_weights.mkdir()

    with pytest.raises(ValueError, match="No xturing.json found"):
        BaseModel.load(str(local_weights))
