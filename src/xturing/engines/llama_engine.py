import math
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine
from xturing.engines.llama_utils import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


class LLamaEngine(CausalEngine):
    config_name: str = "llama_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "sallywww/Llama-7B"
        model = LlamaForCausalLM.from_pretrained(model_name)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(weights_path=weights_path, model=model, tokenizer=tokenizer)

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraEngine(CausalLoraEngine):
    config_name: str = "llama_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "sallywww/Llama-7B"
        model = LlamaForCausalLM.from_pretrained(model_name)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            target_modules=["q_proj", "v_proj"],
        )
