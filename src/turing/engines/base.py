from registry import BaseParent

from .gptj_engine import GPTJEngine
from .llama_engine import LlamaEngine


class BaseEngine(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                "gptj_engine": GPTJEngine,
                "llama_engine": LlamaEngine,
            }
        )
