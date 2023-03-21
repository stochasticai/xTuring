from turing.registry import BaseParent

from turing.engines.gptj_engine import GPTJEngine
from turing.engines.llama_engine import LLamaEngine
from turing.engines.gpt2_engine import GPT2Engine


class BaseEngine(BaseParent):
    registry = {
        GPTJEngine.config_name: GPTJEngine,
        LLamaEngine.config_name: LLamaEngine,
    }
