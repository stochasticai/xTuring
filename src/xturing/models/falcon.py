from typing import Optional

from xturing.engines.falcon_engine import (
    FalconEngine,
    FalconInt8Engine,
    FalconLoraEngine,
    FalconLoraInt8Engine,
    FalconLoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class Falcon(CausalModel):
    config_name: str = "falcon"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FalconEngine.config_name, weights_path)


class FalconLora(CausalLoraModel):
    config_name: str = "falcon_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FalconLoraEngine.config_name, weights_path)


class FalconInt8(CausalInt8Model):
    config_name: str = "falcon_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FalconInt8Engine.config_name, weights_path)


class FalconLoraInt8(CausalLoraInt8Model):
    config_name: str = "falcon_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FalconLoraInt8Engine.config_name, weights_path)


class FalconLoraKbit(CausalLoraKbitModel):
    config_name: str = "falcon_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(FalconLoraKbitEngine.config_name, weights_path)
