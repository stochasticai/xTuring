import tempfile
from pathlib import Path

import pytest

from xturing.config import DEFAULT_DEVICE
from xturing.models import (
    GenericInt8Model,
    GenericLoraInt8Model,
    GenericLoraKbitModel,
    GenericLoraModel,
    GenericModel,
)
from xturing.utils.utils import is_itrex_available


CPU_WITHOUT_ITREX = DEFAULT_DEVICE.type == "cpu" and not is_itrex_available()
NO_CUDA = DEFAULT_DEVICE.type != "cuda"


def test_generic_model():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing_generic"
    model = GenericModel("distilgpt2")
    model.save(str(saving_path))

    model2 = GenericModel(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])


@pytest.mark.skipif(
    CPU_WITHOUT_ITREX,
    reason="GenericInt8Model on CPU requires intel-extension-for-transformers",
)
def test_generic_model_int8():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing_generic_int8"
    model = GenericInt8Model("distilgpt2")
    model.save(str(saving_path))

    model2 = GenericInt8Model(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])


def test_generic_model_lora():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing_generic_lora"
    model = GenericLoraModel("distilgpt2")
    model.save(str(saving_path))

    model2 = GenericLoraModel(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])


@pytest.mark.skipif(NO_CUDA, reason="GenericLoraInt8Model requires CUDA")
def test_generic_model_int8_lora():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing_lora_int8"
    model = GenericLoraInt8Model("distilgpt2")
    model.save(str(saving_path))

    model2 = GenericLoraInt8Model(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])


@pytest.mark.skipif(NO_CUDA, reason="GenericLoraKbitModel requires CUDA")
def test_generic_model_lora_kbit():
    saving_path = Path(tempfile.gettempdir()) / "test_xturing_lora_kbit"
    model = GenericLoraKbitModel("distilgpt2")
    model.save(str(saving_path))

    model2 = GenericLoraKbitModel(str(saving_path))
    model2.generate(texts=["Why are the LLM so important?"])
