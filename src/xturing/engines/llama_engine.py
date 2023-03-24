from pathlib import Path
from typing import Optional, Union

from transformers import LlamaTokenizer

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class LLamaEngine(CausalEngine):
    config_name: str = "llama_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__("/mnt/disks/datadrive/llama_7b_hf", weights_path)

        if weights_path is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "/mnt/disks/datadrive/llama_7b_hf", add_bos_token=False
            )
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.tokenizer = LlamaTokenizer.from_pretrained(weights_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraEngine(CausalLoraEngine):
    config_name: str = "llama_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__("/mnt/disks/datadrive/llama_7b_hf", weights_path)

        if weights_path is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "/mnt/disks/datadrive/llama_7b_hf", add_bos_token=False
            )
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.tokenizer = LlamaTokenizer.from_pretrained(weights_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
