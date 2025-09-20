from typing import Optional

from xturing.engines.gpt_oss_engine import (
    GPTOSS20BEngine,
    GPTOSS20BInt8Engine,
    GPTOSS20BLoraEngine,
    GPTOSS20BLoraInt8Engine,
    GPTOSS20BLoraKbitEngine,
    GPTOSS120BEngine,
    GPTOSS120BInt8Engine,
    GPTOSS120BLoraEngine,
    GPTOSS120BLoraInt8Engine,
    GPTOSS120BLoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class GPTOSS120B(CausalModel):
    config_name: str = "gpt_oss_120b"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS120BEngine.config_name, weights_path)


class GPTOSS120BLora(CausalLoraModel):
    config_name: str = "gpt_oss_120b_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS120BLoraEngine.config_name, weights_path)


class GPTOSS120BInt8(CausalInt8Model):
    config_name: str = "gpt_oss_120b_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS120BInt8Engine.config_name, weights_path)


class GPTOSS120BLoraInt8(CausalLoraInt8Model):
    config_name: str = "gpt_oss_120b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS120BLoraInt8Engine.config_name, weights_path)


class GPTOSS120BLoraKbit(CausalLoraKbitModel):
    config_name: str = "gpt_oss_120b_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS120BLoraKbitEngine.config_name, weights_path)


class GPTOSS20B(CausalModel):
    config_name: str = "gpt_oss_20b"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS20BEngine.config_name, weights_path)


class GPTOSS20BLora(CausalLoraModel):
    config_name: str = "gpt_oss_20b_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS20BLoraEngine.config_name, weights_path)


class GPTOSS20BInt8(CausalInt8Model):
    config_name: str = "gpt_oss_20b_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS20BInt8Engine.config_name, weights_path)


class GPTOSS20BLoraInt8(CausalLoraInt8Model):
    config_name: str = "gpt_oss_20b_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS20BLoraInt8Engine.config_name, weights_path)


class GPTOSS20BLoraKbit(CausalLoraKbitModel):
    config_name: str = "gpt_oss_20b_lora_kbit"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTOSS20BLoraKbitEngine.config_name, weights_path)
