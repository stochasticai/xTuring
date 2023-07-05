import json
from pathlib import Path
from typing import List, Optional, Union

from xturing.engines.generic_engine import (
    GenericEngine,
    GenericInt8Engine,
    GenericLoraEngine,
    GenericLoraInt8Engine,
    GenericLoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class GenericModel(CausalModel):
    config_name: str = "generic"

    def __init__(self, model_name: str, weights_path: Optional[str] = None, **kwargs):
        super().__init__(
            GenericEngine.config_name, weights_path, model_name=model_name, **kwargs
        )

    def _save_config(self, path: Union[str, Path]):
        xturing_config_path = Path(path) / "xturing.json"
        xturing_config = {
            "model_name": self.model_name,
            "engine_name": self.engine.model_name,
            "finetuning_config": self.finetuning_args.dict(),
            "generation_config": self.generation_args.dict(),
        }

        with open(str(xturing_config_path), "w", encoding="utf-8") as f:
            json.dump(xturing_config, f, ensure_ascii=False, indent=4)


class GenericLoraModel(CausalLoraModel):
    config_name: str = "generic_lora"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str] = ["c_attn"],
        weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            GenericLoraEngine.config_name,
            weights_path,
            model_name=model_name,
            target_modules=target_modules,
            **kwargs,
        )


class GenericInt8Model(CausalInt8Model):
    config_name: str = "generic_int8"

    def __init__(self, model_name: str, weights_path: Optional[str] = None, **kwargs):
        super().__init__(
            GenericInt8Engine.config_name, weights_path, model_name=model_name, **kwargs
        )


class GenericLoraInt8Model(CausalLoraInt8Model):
    config_name: str = "generic_lora_int8"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str] = ["c_attn"],
        weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            GenericLoraInt8Engine.config_name,
            weights_path,
            model_name=model_name,
            target_modules=target_modules,
            **kwargs,
        )


class GenericLoraKbitModel(CausalLoraKbitModel):
    config_name: str = "generic_lora_kbit"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str] = ["c_attn"],
        weights_path: Optional[str] = None,
    ):
        super().__init__(
            GenericLoraKbitEngine.config_name,
            weights_path,
            model_name=model_name,
            target_modules=target_modules,
        )
