from typing import Optional

from xturing.engines.mistral_engine import (
    Ministral314BEngine,
    Ministral314BInt8Engine,
    Ministral314BLoraEngine,
    Ministral314BLoraInt8Engine,
    Ministral314BLoraKbitEngine,
    Mistral7BEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class Mistral7B(CausalModel):
    config_name: str = "mistral_7b"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Mistral7BEngine.config_name, weights_path)


class Ministral314B(CausalModel):
    config_name: str = "ministral_3_14b"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Ministral314BEngine.config_name, weights_path)


class Ministral314BLora(CausalLoraModel):
    config_name: str = "ministral_3_14b_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Ministral314BLoraEngine.config_name, weights_path)


class Ministral314BInt8(CausalInt8Model):
    config_name: str = "ministral_3_14b_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Ministral314BInt8Engine.config_name, weights_path)


class Ministral314BLoraInt8(CausalLoraInt8Model):
    config_name: str = "ministral_3_14b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Ministral314BLoraInt8Engine.config_name, weights_path)


class Ministral314BLoraKbit(CausalLoraKbitModel):
    config_name: str = "ministral_3_14b_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(Ministral314BLoraKbitEngine.config_name, weights_path)
