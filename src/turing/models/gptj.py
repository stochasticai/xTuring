from typing import Optional

from turing.engines.gptj_engine import GPTJEngine
from turing.models.causal import CausalModel


class GPTJ(CausalModel):
    config_name: str = "gptj"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJEngine.config_name, weights_path)
