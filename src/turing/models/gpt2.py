from typing import Optional

from turing.engines.gpt2_engine import GPT2Engine

from .causal import CausalModel


class GPT2(CausalModel):
    config_name: str = "gpt2"

    def __init__(
        self, weights_path: Optional[str] = None, engine: str = GPT2Engine.config_name
    ):
        super().__init__(engine, weights_path)
