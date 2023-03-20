from registry import BaseParent

from turing.engines.gptj_engine import GPTJEngine
from turing.engines.llama_engine import LLamaEngine


class BaseEngine(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                GPTJEngine.config_name: GPTJEngine,
                LLamaEngine.config_name: LLamaEngine,
            }
        )
