from typing import Optional

from xturing.engines.llama2_engine import (
    LLama2Engine,
    LLama2Int8Engine,
    LLama2LoraEngine,
    LLama2LoraInt8Engine,
    LLama2LoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class Llama2(CausalModel):
    config_name: str = "llama2"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLama2Engine.config_name, weights_path)


class Llama2Lora(CausalLoraModel):
    config_name: str = "llama2_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLama2LoraEngine.config_name, weights_path)


class Llama2Int8(CausalInt8Model):
    config_name: str = "llama2_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLama2Int8Engine.config_name, weights_path)


class Llama2LoraInt8(CausalLoraInt8Model):
    config_name: str = "llama2_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLama2LoraInt8Engine.config_name, weights_path)


class Llama2LoraKbit(CausalLoraKbitModel):
    config_name: str = "llama2_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLama2LoraKbitEngine.config_name, weights_path)
