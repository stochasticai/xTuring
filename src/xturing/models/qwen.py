from typing import Optional

from xturing.engines.qwen_engine import (
    Qwen3Engine,
    Qwen3Int8Engine,
    Qwen3LoraEngine,
    Qwen3LoraInt8Engine,
    Qwen3LoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class Qwen3(CausalModel):
    config_name: str = "qwen3_0_6b"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Qwen3Engine.config_name, weights_path)


class Qwen3Lora(CausalLoraModel):
    config_name: str = "qwen3_0_6b_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Qwen3LoraEngine.config_name, weights_path)


class Qwen3Int8(CausalInt8Model):
    config_name: str = "qwen3_0_6b_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Qwen3Int8Engine.config_name, weights_path)


class Qwen3LoraInt8(CausalLoraInt8Model):
    config_name: str = "qwen3_0_6b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Qwen3LoraInt8Engine.config_name, weights_path)


class Qwen3LoraKbit(CausalLoraKbitModel):
    config_name: str = "qwen3_0_6b_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Qwen3LoraKbitEngine.config_name, weights_path)
