from typing import Optional

from xturing.engines.bloom_engine import (
    BloomEngine,
    BloomInt8Engine,
    BloomLoraEngine,
    BloomLoraInt8Engine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraModel,
    CausalModel,
)


class Bloom(CausalModel):
    config_name: str = "bloom"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(BloomEngine.config_name, weights_path)


class BloomLora(CausalLoraModel):
    config_name: str = "bloom_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(BloomLoraEngine.config_name, weights_path)


class BloomInt8(CausalInt8Model):
    config_name: str = "bloom_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(BloomInt8Engine.config_name, weights_path)


class BloomLoraInt8(CausalLoraInt8Model):
    config_name: str = "bloom_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(BloomLoraInt8Engine.config_name, weights_path)
