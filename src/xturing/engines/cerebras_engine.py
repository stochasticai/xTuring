from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class CerebrasEngine(CausalEngine):
    config_name: str = "cerebras_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="cerebras/Cerebras-GPT-1.3B", weights_path=weights_path
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class CerebrasLoraEngine(CausalLoraEngine):
    config_name: str = "cerebras_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="cerebras/Cerebras-GPT-1.3B",
            weights_path=weights_path,
            target_modules=["c_attn"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class CerebrasInt8Engine(CausalEngine):
    config_name: str = "cerebras_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="cerebras/Cerebras-GPT-1.3B",
            weights_path=weights_path,
            load_8bit=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class CerebrasLoraInt8Engine(CausalLoraEngine):
    config_name: str = "cerebras_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="cerebras/Cerebras-GPT-1.3B",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["c_attn"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
