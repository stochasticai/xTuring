from typing import Optional

from xturing.engines.gptj_engine import GPTJEngine, GPTJLoraEngine
from xturing.models.causal import CausalLoraModel, CausalModel


class GPTJ(CausalModel):
    config_name: str = "gptj"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJEngine.config_name, weights_path)


class GPTJLORA(CausalLoraModel):
    config_name: str = "gptj_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJLoraEngine.config_name, weights_path)
