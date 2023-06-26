import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import transformers
from torch import nn

from xturing.config.config_data_classes import FinetuningConfig, GenerationConfig
from xturing.config.read_config import load_config, read_yaml
from xturing.engines.causal import CausalEngine, CausalLoraEngine, CausalLoraKbitEngine
from xturing.engines.llama_utils import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from xturing.engines.lora_engine import prepare_model_for_int8_training
from xturing.engines.quant_utils import autotune_warmup, make_quant
from xturing.engines.quant_utils.lrec import get_c4, prepare_models, train_model
from xturing.utils.hub import ModelHub


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


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


class LlamaLoraKbitEngine(CausalLoraKbitEngine):
    config_name: str = "llama_lora_kbit_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "decapoda-research/llama-7b-hf"
        # lrec_config = {
        #     "base_model": model_name,
        #     "intq_checkpoint": str(
        #         Path(__file__).parent / "llama7b-2bit-128g.pt"
        #     ),  ## how to do this
        #     "wbits": wbits,
        #     "lora_target_modules": [
        #         "q_proj",
        #         "v_proj",
        #         "k_proj",
        #         "o_proj",
        #         "up_proj",
        #         "down_proj",
        #         "gate_proj",
        #     ],
        #     # "n_samples": 100,
        #     # "train_cache_dir": "./train_cache/",
        #     # "val_cache_dir": "./val_cache/",
        #     # "ckpt_dir": "./ckpts/",
        #     # "save_dir": "./save/",
        # }

        # # Finetuning config
        # yml_content = read_yaml(
        #     Path(__file__).parent.parent / "config" / "finetuning_config.yaml",
        # )
        # lrec_config.update(yml_content["defaults"])
        # lrec_config.update(yml_content[self.config_name.replace("_engine", "")])

        # model, fp_model = prepare_models(argparse.Namespace(**lrec_config))

        # # The model before applying LoRA
        # self.base_model = fp_model

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            model_name=model_name,
            weights_path=None,
            tokenizer=tokenizer,
            target_modules=["q_proj", "v_proj"],
            load_4bit=True,
        )
