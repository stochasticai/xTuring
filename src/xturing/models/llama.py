from typing import List, Optional, Union

from xturing.engines.llama_engine import (
    LLamaEngine,
    LLamaInt8Engine,
    LlamaLoraEngine,
    LlamaLoraInt8Engine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraModel,
    CausalModel,
)


class Llama(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLamaEngine.config_name, weights_path)


class LlamaLora(CausalLoraModel):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraEngine.config_name, weights_path)


class LlamaInt8(CausalInt8Model):
    config_name: str = "llama_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLamaInt8Engine.config_name, weights_path)


class LlamaLoraInt8(CausalLoraInt8Model):
    config_name: str = "llama_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraInt8Engine.config_name, weights_path)
