from typing import Optional

from xturing.engines.llama_engine import LLama2Engine
from xturing.models.causal import CausalModel


class Llama2(CausalModel):
    config_name: str = "llama2"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLama2Engine.config_name, weights_path)
