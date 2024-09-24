from typing import Optional

from xturing.engines.mixtral_engine import (
    MixtralEngine,
    MixtralInt8Engine,
    MixtralLoraEngine,
    MixtralLoraInt8Engine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraModel,
    CausalModel,
)


class Mixtral(CausalModel):
    config_name: str = "mixtral"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MixtralEngine.config_name, weights_path)


class MixtralLora(CausalLoraModel):
    config_name: str = "mixtral_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MixtralLoraEngine.config_name, weights_path)


class MixtralInt8(CausalInt8Model):
    config_name: str = "mixtral_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MixtralInt8Engine.config_name, weights_path)


class MixtralLoraInt8(CausalLoraInt8Model):
    config_name: str = "mixtral_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MixtralLoraInt8Engine.config_name, weights_path)
