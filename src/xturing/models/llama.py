from typing import List, Optional, Union

from xturing.engines.llama_engine import (
    LLaMAEngine,
    LLaMAInt8Engine,
    LLaMALoraEngine,
    LLaMALoraInt8Engine,
)
from xturing.models.causal import CausalLoraModel, CausalModel


class LLaMA(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLaMAEngine.config_name, weights_path)


class LLaMALORA(CausalLoraModel):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLaMALoraEngine.config_name, weights_path)


class LLaMAInt8(CausalModel):
    config_name: str = "llama_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLaMAInt8Engine.config_name, weights_path)


class LLaMALORAInt8(CausalLoraModel):
    config_name: str = "llama_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLaMALoraInt8Engine.config_name, weights_path)
