from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class BloomEngine(CausalEngine):
    config_name: str = "bloom_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="bigscience/bloom-1b1", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class BloomLoraEngine(CausalLoraEngine):
    config_name: str = "bloom_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="bigscience/bloom-1b1",
            weights_path=weights_path,
            target_modules=["query_key_value"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class BloomInt8Engine(CausalEngine):
    config_name: str = "bloom_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="bigscience/bloom-1b1",
            weights_path=weights_path,
            load_8bit=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class BloomLoraInt8Engine(CausalLoraEngine):
    config_name: str = "bloom_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="bigscience/bloom-1b1",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["query_key_value"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
