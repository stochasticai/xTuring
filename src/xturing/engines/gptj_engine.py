from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class GPTJEngine(CausalEngine):
    config_name: str = "gptj_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="EleutherAI/gpt-j-6B", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPTJLoraEngine(CausalLoraEngine):
    config_name: str = "gptj_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="EleutherAI/gpt-j-6B", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
