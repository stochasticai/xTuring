from pathlib import Path
from typing import Optional, Union

from turing.engines.causal import CausalEngine


class LLamaEngine(CausalEngine):
    config_name: str = "llama_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__("decapoda-research/llama-7b-hf", weights_path)

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraEngine(LLamaEngine):
    config_name: str = "llama_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__("decapoda-research/llama-7b-hf", weights_path)
