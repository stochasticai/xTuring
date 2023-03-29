from typing import Optional

from xturing.engines.cerebras_engine import (
    CerebrasEngine,
    CerebrasInt8Engine,
    CerebrasLoraEngine,
    CerebrasLoraInt8Engine,
)
from xturing.models.causal import CausalLoraModel, CausalModel


class Cerebras(CausalModel):
    config_name: str = "cerebras"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(CerebrasEngine.config_name, weights_path)


class CerebrasLora(CausalLoraModel):
    config_name: str = "cerebras_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(CerebrasLoraEngine.config_name, weights_path)


class CerebrasInt8(CausalModel):
    config_name: str = "cerebras_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(CerebrasInt8Engine.config_name, weights_path)


class CerebrasLoraInt8(CausalLoraModel):
    config_name: str = "cerebras_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(CerebrasLoraInt8Engine.config_name, weights_path)
