from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class GalacticaEngine(CausalEngine):
    config_name: str = "galactica_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/galactica-6.7b", weights_path=weights_path
        )

        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 1


class GalacticaLoraEngine(CausalLoraEngine):
    config_name: str = "galactica_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/galactica-6.7b",
            weights_path=weights_path,
            target_modules=["q_proj", "v_proj"],
        )

        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 1


class GalacticaInt8Engine(CausalEngine):
    config_name: str = "galactica_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/galactica-6.7b",
            weights_path=weights_path,
            load_8bit=True,
        )
        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 1


class GalacticaLoraInt8Engine(CausalLoraEngine):
    config_name: str = "galactica_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/galactica-6.7b",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
        )

        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 1
