import os
from pathlib import Path
from typing import Optional, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from xturing.engines.causal import CausalEngine, CausalLoraEngine
from xturing.engines.gptj_utils.gptj import GPTJAttention
from xturing.engines.lora_engine import prepare_model_for_int8_training


class GenericEngine(CausalEngine):
    config_name: str = "generic_engine"

    def __init__(self, model_name, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name=model_name, weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
