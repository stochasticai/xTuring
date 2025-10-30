from xturing.models import BaseModel

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}


def test_minimax_m2_model_creation():
    """Test that MiniMaxM2 models can be created from registry."""
    # Test all variants
    models_to_test = [
        "minimax_m2",
        "minimax_m2_lora",
        "minimax_m2_int8",
        "minimax_m2_lora_int8",
        "minimax_m2_lora_kbit",
    ]

    for model_name in models_to_test:
        # Test that the model is registered
        assert (
            model_name in BaseModel.registry
        ), f"Model {model_name} not found in registry"

        # Test that the model class can be instantiated (without actually loading weights)
        model_class = BaseModel.registry[model_name]
        assert model_class is not None, f"Model class for {model_name} is None"


def test_minimax_m2_engine_class_attributes():
    """Test that MiniMaxM2 engine classes have correct attributes."""
    from xturing.engines.minimax_m2_engine import MiniMaxM2Engine, MiniMaxM2LoraEngine

    # Test engine config names
    assert MiniMaxM2Engine.config_name == "minimax_m2_engine"
    assert MiniMaxM2LoraEngine.config_name == "minimax_m2_lora_engine"


def test_minimax_m2_model_class_attributes():
    """Test that MiniMaxM2 model classes have correct attributes."""
    from xturing.models.minimax_m2 import MiniMaxM2, MiniMaxM2Lora

    # Test model config names
    assert MiniMaxM2.config_name == "minimax_m2"
    assert MiniMaxM2Lora.config_name == "minimax_m2_lora"


def test_minimax_m2_config_values():
    """Test that configuration values are properly loaded for MiniMaxM2 models."""
    from pathlib import Path

    from xturing.config.read_config import read_yaml

    # Test that our config entries exist
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/generation_config.yaml"
    )
    yml_content = read_yaml(str(config_path))

    # Check that our MiniMaxM2 configs exist
    assert "minimax_m2" in yml_content
    assert yml_content["minimax_m2"]["max_new_tokens"] == 512

    # Test finetuning config
    finetuning_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "src/xturing/config/finetuning_config.yaml"
    )
    finetuning_yml_content = read_yaml(str(finetuning_config_path))

    assert "minimax_m2" in finetuning_yml_content
    assert finetuning_yml_content["minimax_m2"]["max_length"] == 2048


def test_minimax_m2_engine_registry():
    """Test that MiniMaxM2 engines are properly registered."""
    from xturing.engines.base import BaseEngine

    # Check all engines are registered
    engine_names = [
        "minimax_m2_engine",
        "minimax_m2_lora_engine",
        "minimax_m2_int8_engine",
        "minimax_m2_lora_int8_engine",
        "minimax_m2_lora_kbit_engine",
    ]

    for engine_name in engine_names:
        assert (
            engine_name in BaseEngine.registry
        ), f"Engine {engine_name} not found in registry"
