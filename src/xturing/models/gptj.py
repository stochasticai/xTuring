from typing import Optional

from xturing.engines.gptj_engine import (
    GPTJEngine,
    GPTJInt8Engine,
    GPTJLoraEngine,
    GPTJLoraInt8Engine,
)
from xturing.models.causal import CausalLoraModel, CausalModel


class GPTJ(CausalModel):
    config_name: str = "gptj"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJEngine.config_name, weights_path)


class GPTJLORA(CausalLoraModel):
    config_name: str = "gptj_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJLoraEngine.config_name, weights_path)


class GPTJInt8(CausalModel):
    config_name: str = "gptj_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJInt8Engine.config_name, weights_path)


class GPTJLORAInt8(CausalLoraModel):
    config_name: str = "gptj_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJLoraInt8Engine.config_name, weights_path)
