from typing import Optional

from xturing.engines.opt_engine import (
    OPTEngine,
    OPTInt8Engine,
    OPTLoraEngine,
    OPTLoraInt8Engine,
)
from xturing.models.causal import CausalLoraModel, CausalModel


class OPT(CausalModel):
    config_name: str = "opt"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(OPTEngine.config_name, weights_path)


class OPTLora(CausalLoraModel):
    config_name: str = "opt_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(OPTLoraEngine.config_name, weights_path)


class OPTInt8(CausalModel):
    config_name: str = "opt_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(OPTInt8Engine.config_name, weights_path)


class OPTLoraInt8(CausalLoraModel):
    config_name: str = "opt_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(OPTLoraInt8Engine.config_name, weights_path)
