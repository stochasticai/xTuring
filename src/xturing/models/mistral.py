from typing import Optional

from xturing.engines.mistral_engine import (
    MistralEngine,
    MistralInt8Engine,
    MistralLoraEngine,
    MistralLoraInt8Engine,
    MistralLoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class Mistral(CausalModel):
    config_name: str = "mistral"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MistralEngine.config_name, weights_path)


class MistralLora(CausalLoraModel):
    config_name: str = "mistral_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MistralLoraEngine.config_name, weights_path)


class MistralInt8(CausalInt8Model):
    config_name: str = "mistral_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MistralInt8Engine.config_name, weights_path)


class MistralLoraInt8(CausalLoraInt8Model):
    config_name: str = "mistral_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MistralLoraInt8Engine.config_name, weights_path)


class MistralLoraKbit(CausalLoraKbitModel):
    config_name: str = "mistral_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MistralLoraKbitEngine.config_name, weights_path)
