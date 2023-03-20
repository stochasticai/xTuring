from typing import Union, Optional
from pathlib import Path
from transformers import LlamaForCausalLM

class LLamaEngine:
    def __init__(
        self, 
        weights_path: Optional[Union[str, Path]] = None
    ):
        if weights_path is None:
            self.model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        else:
            assert Path(weights_path).is_dir(), "The weights path should be a existing directory"
            self.model = LlamaForCausalLM.from_pretrained(weights_path)
