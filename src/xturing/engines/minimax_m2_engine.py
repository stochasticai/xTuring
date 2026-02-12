from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class MiniMaxM2Engine(CausalEngine):
    config_name: str = "minimax_m2_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "MiniMaxAI/MiniMax-M2"
        super().__init__(
            model_name=model_name, weights_path=weights_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class MiniMaxM2LoraEngine(CausalLoraEngine):
    config_name: str = "minimax_m2_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "MiniMaxAI/MiniMax-M2"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class MiniMaxM2Int8Engine(CausalEngine):
    config_name: str = "minimax_m2_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "MiniMaxAI/MiniMax-M2"
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class MiniMaxM2LoraInt8Engine(CausalLoraEngine):
    config_name: str = "minimax_m2_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "MiniMaxAI/MiniMax-M2"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class MiniMaxM2LoraKbitEngine(CausalLoraEngine):
    config_name: str = "minimax_m2_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "MiniMaxAI/MiniMax-M2"
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_4bit=True,
            target_modules=target_modules,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
