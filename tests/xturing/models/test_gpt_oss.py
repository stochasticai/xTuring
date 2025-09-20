from xturing.models import BaseModel

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}


def test_gpt_oss_120b_model_creation():
    """Test that GPT-OSS 120B models can be created from registry."""
    # Test all variants
    models_to_test = [
        "gpt_oss_120b",
        "gpt_oss_120b_lora",
        "gpt_oss_120b_int8",
        "gpt_oss_120b_lora_int8",
        "gpt_oss_120b_lora_kbit",
    ]

    for model_name in models_to_test:
        # Test that the model is registered
        assert (
            model_name in BaseModel.registry
        ), f"Model {model_name} not found in registry"

        # Test that the model class can be instantiated (without actually loading weights)
        model_class = BaseModel.registry[model_name]
        assert model_class is not None, f"Model class for {model_name} is None"


def test_gpt_oss_20b_model_creation():
    """Test that GPT-OSS 20B models can be created from registry."""
    # Test all variants
    models_to_test = [
        "gpt_oss_20b",
        "gpt_oss_20b_lora",
        "gpt_oss_20b_int8",
        "gpt_oss_20b_lora_int8",
        "gpt_oss_20b_lora_kbit",
    ]

    for model_name in models_to_test:
        # Test that the model is registered
        assert (
            model_name in BaseModel.registry
        ), f"Model {model_name} not found in registry"

        # Test that the model class can be instantiated (without actually loading weights)
        model_class = BaseModel.registry[model_name]
        assert model_class is not None, f"Model class for {model_name} is None"


def test_gpt_oss_engine_class_attributes():
    """Test that GPT-OSS engine classes have correct attributes."""
    from xturing.engines.gpt_oss_engine import (
        GPTOSS20BEngine,
        GPTOSS20BLoraEngine,
        GPTOSS120BEngine,
        GPTOSS120BLoraEngine,
    )

    # Test engine config names
    assert GPTOSS120BEngine.config_name == "gpt_oss_120b_engine"
    assert GPTOSS20BEngine.config_name == "gpt_oss_20b_engine"
    assert GPTOSS120BLoraEngine.config_name == "gpt_oss_120b_lora_engine"
    assert GPTOSS20BLoraEngine.config_name == "gpt_oss_20b_lora_engine"


def test_gpt_oss_model_class_attributes():
    """Test that GPT-OSS model classes have correct attributes."""
    from xturing.models.gpt_oss import (
        GPTOSS20B,
        GPTOSS120B,
        GPTOSS20BLora,
        GPTOSS120BLora,
    )

    # Test model config names
    assert GPTOSS120B.config_name == "gpt_oss_120b"
    assert GPTOSS20B.config_name == "gpt_oss_20b"
    assert GPTOSS120BLora.config_name == "gpt_oss_120b_lora"
    assert GPTOSS20BLora.config_name == "gpt_oss_20b_lora"


def test_gpt_oss_harmony_format_helper():
    """Test the harmony format setup helper function."""
    from unittest.mock import MagicMock

    from xturing.engines.gpt_oss_engine import setup_harmony_format

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.chat_template = None

    # Call setup function
    setup_harmony_format(mock_tokenizer)

    # Check that pad_token was set
    assert mock_tokenizer.pad_token == "</s>"

    # Check that chat_template was set for harmony format
    assert mock_tokenizer.chat_template is not None
    assert "<|user|>" in mock_tokenizer.chat_template
    assert "<|assistant|>" in mock_tokenizer.chat_template


def test_gpt_oss_config_values():
    """Test that configuration values are properly loaded for GPT-OSS models."""
    from pathlib import Path

    from xturing.config.read_config import read_yaml

    # Test that our config entries exist
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/generation_config.yaml"
    )
    yml_content = read_yaml(str(config_path))

    # Check that our GPT-OSS configs exist
    assert "gpt_oss_120b" in yml_content
    assert "gpt_oss_20b" in yml_content
    assert yml_content["gpt_oss_120b"]["max_new_tokens"] == 512
    assert yml_content["gpt_oss_20b"]["max_new_tokens"] == 512

    # Test finetuning config
    finetuning_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/finetuning_config.yaml"
    )
    finetuning_yml_content = read_yaml(str(finetuning_config_path))

    assert "gpt_oss_120b" in finetuning_yml_content
    assert "gpt_oss_20b" in finetuning_yml_content
    assert finetuning_yml_content["gpt_oss_120b"]["max_length"] == 2048
    assert finetuning_yml_content["gpt_oss_20b"]["max_length"] == 2048


def test_gpt_oss_engine_registry():
    """Test that GPT-OSS engines are properly registered."""
    from xturing.engines.base import BaseEngine

    # Check all engines are registered
    engine_names = [
        "gpt_oss_120b_engine",
        "gpt_oss_120b_lora_engine",
        "gpt_oss_120b_int8_engine",
        "gpt_oss_120b_lora_int8_engine",
        "gpt_oss_120b_lora_kbit_engine",
        "gpt_oss_20b_engine",
        "gpt_oss_20b_lora_engine",
        "gpt_oss_20b_int8_engine",
        "gpt_oss_20b_lora_int8_engine",
        "gpt_oss_20b_lora_kbit_engine",
    ]

    for engine_name in engine_names:
        assert (
            engine_name in BaseEngine.registry
        ), f"Engine {engine_name} not found in registry"
