import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from xturing.engines.causal import CausalEngine, CausalLoraEngine
from xturing.engines.llama_utils import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from xturing.engines.lora_engine import prepare_model_for_int8_training


class LLamaEngine(CausalEngine):
    config_name: str = "llama_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(weights_path=weights_path, model=model, tokenizer=tokenizer)

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraEngine(CausalLoraEngine):
    config_name: str = "llama_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            target_modules=["q_proj", "v_proj"],
        )


class LLamaInt8Engine(CausalEngine):
    config_name: str = "llama_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path, model=model, tokenizer=tokenizer, load_8bit=True
        )

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraInt8Engine(CausalLoraEngine):
    config_name: str = "llama_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model)

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
        )
