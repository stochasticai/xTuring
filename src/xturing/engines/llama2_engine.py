from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine, CausalLoraKbitEngine


class LLama2Engine(CausalEngine):
    config_name: str = "llama2_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="daryl149/llama-2-7b-chat-hf",
            weights_path=weights_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class LLama2LoraEngine(CausalLoraEngine):
    config_name: str = "llama2_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="daryl149/llama-2-7b-chat-hf",
            weights_path=weights_path,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class LLama2Int8Engine(CausalEngine):
    config_name: str = "llama2_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="daryl149/llama-2-7b-chat-hf",
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class LLama2LoraInt8Engine(CausalLoraEngine):
    config_name: str = "llama2_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="daryl149/llama-2-7b-chat-hf",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class LLama2LoraKbitEngine(CausalLoraKbitEngine):
    config_name: str = "llama2_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "daryl149/llama-2-7b-chat-hf"
        super().__init__(
            model_name=model_name,
            weights_path=None,
            target_modules=["q_proj", "v_proj"],
            trust_remote_code=True,
            load_4bit=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
