from typing import Union, Optional
from pathlib import Path
from transformers import GPTJForCausalLM

class GPTJEngine:
    def __init__(
        self, 
        weights_path: Optional[Union[str, Path]] = None
    ):
        if weights_path is None:
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            assert Path(weights_path).is_dir(), "The weights path should be a existing directory"
            self.model = GPTJForCausalLM.from_pretrained(weights_path)