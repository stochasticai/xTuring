from typing import List, Optional

from xturing.engines.generic_engine import (
    GenericEngine,
    GenericInt8Engine,
    GenericLoraEngine,
    GenericLoraInt8Engine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraModel,
    CausalModel,
)


class GenericModel(CausalModel):
    config_name: str = "generic"

    def __init__(
        self,
        model_name: str,
        weights_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            GenericEngine.config_name,
            weights_path,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )


class GenericLoraModel(CausalLoraModel):
    config_name: str = "generic_lora"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str] = ["c_attn"],
        weights_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            GenericLoraEngine.config_name,
            weights_path,
            model_name=model_name,
            target_modules=target_modules,
            trust_remote_code=trust_remote_code,
        )


class GenericInt8Model(CausalInt8Model):
    config_name: str = "generic_int8"

    def __init__(
        self,
        model_name: str,
        weights_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            GenericInt8Engine.config_name,
            weights_path,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )


class GenericLoraInt8Model(CausalLoraInt8Model):
    config_name: str = "generic_lora_int8"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str] = ["c_attn"],
        weights_path: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            GenericLoraInt8Engine.config_name,
            weights_path,
            model_name=model_name,
            target_modules=target_modules,
            trust_remote_code=trust_remote_code,
        )
