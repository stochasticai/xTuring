from turing.registry import BaseParent

from turing.engines.gptj_engine import GPTJEngine
from turing.engines.llama_engine import LLamaEngine
from turing.engines.gpt2_engine import GPT2Engine


class BaseEngine(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                "gptj_engine": GPTJEngine,
                "llama_engine": LLamaEngine,
                "gpt2_engine": GPT2Engine
            }
        )
