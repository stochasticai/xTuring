from typing import Optional

from turing.engines.gpt2_engine import GPT2Engine

from .causal import CausalLoraModel, CausalModel


class GPT2(CausalModel):
    config_name: str = "gpt2"

    def __init__(
        self, weights_path: Optional[str] = None, engine: str = GPT2Engine.config_name
    ):
        super().__init__(engine, weights_path)


class GPT2LORA(CausalLoraModel):
    config_name: str = "gpt2_lora"

    def __init__(
        self, weights_path: Optional[str] = None, engine: str = GPT2Engine.config_name
    ):
        super().__init__(engine, weights_path)
