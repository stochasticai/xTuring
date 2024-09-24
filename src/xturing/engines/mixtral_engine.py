from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class MixtralEngine(CausalEngine):
    config_name: str = "mixtral_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="mistral-community/Mixtral-8x22B-v0.1",
            weights_path=weights_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class MixtralLoraEngine(CausalLoraEngine):
    config_name: str = "mixtral_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="mistral-community/Mixtral-8x22B-v0.1",
            weights_path=weights_path,
            target_modules=["q_proj", "v_proj"],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class MixtralInt8Engine(CausalEngine):
    config_name: str = "mixtral_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="mistral-community/Mixtral-8x22B-v0.1",
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class MixtralLoraInt8Engine(CausalLoraEngine):
    config_name: str = "mixtral_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="mistral-community/Mixtral-8x22B-v0.1",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
