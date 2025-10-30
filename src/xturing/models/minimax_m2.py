from typing import Optional

from xturing.engines.minimax_m2_engine import (
    MiniMaxM2Engine,
    MiniMaxM2Int8Engine,
    MiniMaxM2LoraEngine,
    MiniMaxM2LoraInt8Engine,
    MiniMaxM2LoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class MiniMaxM2(CausalModel):
    config_name: str = "minimax_m2"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MiniMaxM2Engine.config_name, weights_path)


class MiniMaxM2Lora(CausalLoraModel):
    config_name: str = "minimax_m2_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MiniMaxM2LoraEngine.config_name, weights_path)


class MiniMaxM2Int8(CausalInt8Model):
    config_name: str = "minimax_m2_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MiniMaxM2Int8Engine.config_name, weights_path)


class MiniMaxM2LoraInt8(CausalLoraInt8Model):
    config_name: str = "minimax_m2_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MiniMaxM2LoraInt8Engine.config_name, weights_path)


class MiniMaxM2LoraKbit(CausalLoraKbitModel):
    config_name: str = "minimax_m2_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MiniMaxM2LoraKbitEngine.config_name, weights_path)
