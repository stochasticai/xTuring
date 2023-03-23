from typing import Optional

from xturing.engines.gpt2_engine import GPT2Engine, GPT2LoraEngine

from .causal import CausalLoraModel, CausalModel


class GPT2(CausalModel):
    config_name: str = "gpt2"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPT2Engine.config_name, weights_path)


class GPT2LORA(CausalLoraModel):
    config_name: str = "gpt2_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPT2LoraEngine.config_name, weights_path)
