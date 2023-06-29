from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine, CausalLoraKbitEngine


class FalconEngine(CausalEngine):
    config_name: str = "falcon_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="tiiuae/falcon-7b",
            weights_path=weights_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class FalconLoraEngine(CausalLoraEngine):
    config_name: str = "falcon_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="tiiuae/falcon-7b",
            weights_path=weights_path,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class FalconInt8Engine(CausalEngine):
    config_name: str = "falcon_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="tiiuae/falcon-7b",
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class FalconLoraInt8Engine(CausalLoraEngine):
    config_name: str = "falcon_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="tiiuae/falcon-7b",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class FalconLoraKbitEngine(CausalLoraKbitEngine):
    config_name: str = "falcon_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "tiiuae/falcon-7b"
        super().__init__(
            model_name=model_name,
            weights_path=None,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            trust_remote_code=True,
            load_4bit=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
