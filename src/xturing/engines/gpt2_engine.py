from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class GPT2Engine(CausalEngine):
    config_name: str = "gpt2_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="gpt2", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPT2LoraEngine(CausalLoraEngine):
    config_name: str = "gpt2_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="gpt2", weights_path=weights_path, target_modules=["c_attn"]
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPT2Int8Engine(CausalEngine):
    config_name: str = "gpt2_engine_int8"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="gpt2", weights_path=weights_path, load_8bit=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPT2LoraInt8Engine(CausalLoraEngine):
    config_name: str = "gpt2_lora_engine_int8"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="gpt2",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["c_attn"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
