import os
from pathlib import Path
from typing import Optional, Union

import torch
import transformers
from peft import prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class GPTJEngine(CausalEngine):
    config_name: str = "gptj_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="EleutherAI/gpt-j-6B", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPTJLoraEngine(CausalLoraEngine):
    config_name: str = "gptj_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="EleutherAI/gpt-j-6B", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token


class GPTJInt8Engine(CausalEngine):
    config_name: str = "gptj_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", load_in_8bit=True, device_map=device_map
        )
        # add_adapters(model)

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-j-6B", load_in_8bit=True, device_map=device_map
        )
        tokenizer.pad_token = self.tokenizer.eos_token
        super().__init__(
            weights_path=weights_path, model=model, tokenizer=tokenizer, load_8bit=True
        )


class GPTJLoraInt8Engine(CausalLoraEngine):
    config_name: str = "gptj_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        # transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", load_in_8bit=True, device_map=device_map
        )

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        tokenizer.pad_token = tokenizer.eos_token
        for param in model.parameters():
            param.data = param.data.contiguous()
        model = prepare_model_for_int8_training(
            model, output_embedding_layer_name="lm_head"
        )

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
        )
