import os
from pathlib import Path
from typing import Optional, Union

from transformers import AutoTokenizer, MambaForCausalLM

from xturing.engines.causal import CausalEngine

class MambaEngine(CausalEngine):
    config_name: str = "mamba_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "state-spaces/mamba-2.8b-hf"
        model = MambaForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        super().__init__(weights_path=weights_path, model=model, tokenizer=tokenizer)


    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)
