from pathlib import Path

from xturing.config.read_config import read_yaml
from xturing.engines.qwen_engine import (
    Qwen3Engine,
    Qwen3Int8Engine,
    Qwen3LoraEngine,
    Qwen3LoraInt8Engine,
    Qwen3LoraKbitEngine,
)
from xturing.models import BaseModel
from xturing.models.qwen import (
    Qwen3,
    Qwen3Int8,
    Qwen3Lora,
    Qwen3LoraInt8,
    Qwen3LoraKbit,
)


def test_qwen3_model_registry_entries_present():
    model_names = [
        "qwen3_0_6b",
        "qwen3_0_6b_lora",
        "qwen3_0_6b_int8",
        "qwen3_0_6b_lora_int8",
        "qwen3_0_6b_lora_kbit",
    ]

    for model_name in model_names:
        assert model_name in BaseModel.registry
        assert BaseModel.registry[model_name] is not None


def test_qwen3_model_class_config_names():
    assert Qwen3.config_name == "qwen3_0_6b"
    assert Qwen3Lora.config_name == "qwen3_0_6b_lora"
    assert Qwen3Int8.config_name == "qwen3_0_6b_int8"
    assert Qwen3LoraInt8.config_name == "qwen3_0_6b_lora_int8"
    assert Qwen3LoraKbit.config_name == "qwen3_0_6b_lora_kbit"


def test_qwen3_engine_class_config_names():
    assert Qwen3Engine.config_name == "qwen3_0_6b_engine"
    assert Qwen3LoraEngine.config_name == "qwen3_0_6b_lora_engine"
    assert Qwen3Int8Engine.config_name == "qwen3_0_6b_int8_engine"
    assert Qwen3LoraInt8Engine.config_name == "qwen3_0_6b_lora_int8_engine"
    assert Qwen3LoraKbitEngine.config_name == "qwen3_0_6b_lora_kbit_engine"


def test_qwen3_config_entries_exist():
    config_dir = (
        Path(__file__).parent.parent.parent.parent / "src" / "xturing" / "config"
    )

    generation_config = read_yaml(str(config_dir / "generation_config.yaml"))
    assert "qwen3_0_6b" in generation_config
    assert "qwen3_0_6b_lora" in generation_config
    assert "qwen3_0_6b_int8" in generation_config
    assert "qwen3_0_6b_lora_int8" in generation_config
    assert "qwen3_0_6b_lora_kbit" in generation_config

    finetuning_config = read_yaml(str(config_dir / "finetuning_config.yaml"))
    assert "qwen3_0_6b" in finetuning_config
    assert "qwen3_0_6b_lora" in finetuning_config
    assert "qwen3_0_6b_int8" in finetuning_config
    assert "qwen3_0_6b_lora_int8" in finetuning_config
    assert "qwen3_0_6b_lora_kbit" in finetuning_config
