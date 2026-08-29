from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine, CausalLoraKbitEngine

_MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
_MINISTRAL_MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"
_MINISTRAL_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class Mistral7BEngine(CausalEngine):
    config_name: str = "mistral_7b_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_MISTRAL_MODEL_NAME,
            weights_path=weights_path,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Ministral314BEngine(CausalEngine):
    config_name: str = "ministral_3_14b_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_MINISTRAL_MODEL_NAME,
            weights_path=weights_path,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Ministral314BLoraEngine(CausalLoraEngine):
    config_name: str = "ministral_3_14b_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_MINISTRAL_MODEL_NAME,
            weights_path=weights_path,
            target_modules=_MINISTRAL_TARGET_MODULES,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Ministral314BInt8Engine(CausalEngine):
    config_name: str = "ministral_3_14b_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_MINISTRAL_MODEL_NAME,
            weights_path=weights_path,
            load_8bit=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Ministral314BLoraInt8Engine(CausalLoraEngine):
    config_name: str = "ministral_3_14b_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_MINISTRAL_MODEL_NAME,
            weights_path=weights_path,
            load_8bit=True,
            target_modules=_MINISTRAL_TARGET_MODULES,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Ministral314BLoraKbitEngine(CausalLoraKbitEngine):
    config_name: str = "ministral_3_14b_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_MINISTRAL_MODEL_NAME,
            weights_path=weights_path,
            target_modules=_MINISTRAL_TARGET_MODULES,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
