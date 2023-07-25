from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine


class LLama2Engine(CausalEngine):
    config_name: str = "llama2_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="daryl149/llama-2-7b-chat-hf",
            weights_path=weights_path,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
