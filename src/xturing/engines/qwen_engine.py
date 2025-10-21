from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine, CausalLoraKbitEngine

_DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class Qwen3Engine(CausalEngine):
    config_name: str = "qwen3_0_6b_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_DEFAULT_MODEL_NAME,
            weights_path=weights_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Qwen3LoraEngine(CausalLoraEngine):
    config_name: str = "qwen3_0_6b_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_DEFAULT_MODEL_NAME,
            weights_path=weights_path,
            target_modules=_TARGET_MODULES,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Qwen3Int8Engine(CausalEngine):
    config_name: str = "qwen3_0_6b_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_DEFAULT_MODEL_NAME,
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Qwen3LoraInt8Engine(CausalLoraEngine):
    config_name: str = "qwen3_0_6b_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_DEFAULT_MODEL_NAME,
            weights_path=weights_path,
            load_8bit=True,
            target_modules=_TARGET_MODULES,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class Qwen3LoraKbitEngine(CausalLoraKbitEngine):
    config_name: str = "qwen3_0_6b_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name=_DEFAULT_MODEL_NAME,
            weights_path=weights_path,
            target_modules=_TARGET_MODULES,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
