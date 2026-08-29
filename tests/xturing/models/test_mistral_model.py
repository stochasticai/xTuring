from xturing.models import BaseModel


def test_mistral_7b_model_creation():
    assert "mistral_7b" in BaseModel.registry
    assert BaseModel.registry["mistral_7b"] is not None


def test_mistral_7b_engine_class_attributes():
    from xturing.engines.mistral_engine import Mistral7BEngine

    assert Mistral7BEngine.config_name == "mistral_7b_engine"


def test_mistral_7b_model_class_attributes():
    from xturing.models.mistral import Mistral7B

    assert Mistral7B.config_name == "mistral_7b"


def test_mistral_7b_config_values():
    from pathlib import Path

    from xturing.config.read_config import read_yaml

    generation_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/generation_config.yaml"
    )
    generation_yml_content = read_yaml(str(generation_config_path))
    assert "mistral_7b" in generation_yml_content
    assert generation_yml_content["mistral_7b"]["max_new_tokens"] == 256

    finetuning_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/finetuning_config.yaml"
    )
    finetuning_yml_content = read_yaml(str(finetuning_config_path))
    assert "mistral_7b" in finetuning_yml_content
    assert finetuning_yml_content["mistral_7b"]["optimizer_name"] == "cpu_adam"


def test_mistral_7b_engine_registry():
    from xturing.engines.base import BaseEngine

    assert "mistral_7b_engine" in BaseEngine.registry


def test_ministral_3_14b_model_creation():
    model_names = [
        "ministral_3_14b",
        "ministral_3_14b_lora",
        "ministral_3_14b_int8",
        "ministral_3_14b_lora_int8",
        "ministral_3_14b_lora_kbit",
    ]
    for model_name in model_names:
        assert model_name in BaseModel.registry
        assert BaseModel.registry[model_name] is not None


def test_ministral_3_14b_engine_class_attributes():
    from xturing.engines.mistral_engine import (
        Ministral314BEngine,
        Ministral314BInt8Engine,
        Ministral314BLoraEngine,
        Ministral314BLoraInt8Engine,
        Ministral314BLoraKbitEngine,
    )

    assert Ministral314BEngine.config_name == "ministral_3_14b_engine"
    assert Ministral314BLoraEngine.config_name == "ministral_3_14b_lora_engine"
    assert Ministral314BInt8Engine.config_name == "ministral_3_14b_int8_engine"
    assert Ministral314BLoraInt8Engine.config_name == "ministral_3_14b_lora_int8_engine"
    assert Ministral314BLoraKbitEngine.config_name == "ministral_3_14b_lora_kbit_engine"


def test_ministral_3_14b_model_class_attributes():
    from xturing.models.mistral import (
        Ministral314B,
        Ministral314BInt8,
        Ministral314BLora,
        Ministral314BLoraInt8,
        Ministral314BLoraKbit,
    )

    assert Ministral314B.config_name == "ministral_3_14b"
    assert Ministral314BLora.config_name == "ministral_3_14b_lora"
    assert Ministral314BInt8.config_name == "ministral_3_14b_int8"
    assert Ministral314BLoraInt8.config_name == "ministral_3_14b_lora_int8"
    assert Ministral314BLoraKbit.config_name == "ministral_3_14b_lora_kbit"


def test_ministral_3_14b_config_values():
    from pathlib import Path

    from xturing.config.read_config import read_yaml

    generation_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/generation_config.yaml"
    )
    generation_yml_content = read_yaml(str(generation_config_path))
    assert "ministral_3_14b" in generation_yml_content
    assert generation_yml_content["ministral_3_14b"]["max_new_tokens"] == 512

    finetuning_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/finetuning_config.yaml"
    )
    finetuning_yml_content = read_yaml(str(finetuning_config_path))
    assert "ministral_3_14b" in finetuning_yml_content
    assert finetuning_yml_content["ministral_3_14b"]["max_length"] == 2048


def test_ministral_3_14b_engine_registry():
    from xturing.engines.base import BaseEngine

    engine_names = [
        "ministral_3_14b_engine",
        "ministral_3_14b_lora_engine",
        "ministral_3_14b_int8_engine",
        "ministral_3_14b_lora_int8_engine",
        "ministral_3_14b_lora_kbit_engine",
    ]
    for engine_name in engine_names:
        assert engine_name in BaseEngine.registry
