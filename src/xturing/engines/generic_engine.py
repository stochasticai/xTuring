import os
from pathlib import Path
from typing import List, Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class GenericEngine(CausalEngine):
    config_name: str = "generic_engine"

    def __init__(
        self,
        model_name: str,
        weights_path: Optional[Union[str, Path]] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            trust_remote_code=trust_remote_code,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GenericLoraEngine(CausalLoraEngine):
    config_name: str = "generic_lora_engine"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str],
        weights_path: Optional[Union[str, Path]] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            target_modules=target_modules,
            trust_remote_code=trust_remote_code,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GenericInt8Engine(CausalEngine):
    config_name: str = "generic_engine_int8"

    def __init__(
        self,
        model_name,
        weights_path: Optional[Union[str, Path]] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            trust_remote_code=trust_remote_code,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GenericLoraInt8Engine(CausalLoraEngine):
    config_name: str = "generic_lora_engine_int8"

    def __init__(
        self,
        model_name: str,
        target_modules: List[str],
        weights_path: Optional[Union[str, Path]] = None,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__(
            model_name=model_name,
            weights_path=weights_path,
            load_8bit=True,
            target_modules=target_modules,
            trust_remote_code=trust_remote_code,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
