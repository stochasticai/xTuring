from turing.engines.gpt2_engine import GPT2Engine
from turing.engines.gptj_engine import GPTJEngine
from turing.engines.llama_engine import LLamaEngine
from turing.registry import BaseParent


class BaseEngine(BaseParent):
    registry = {
        GPTJEngine.config_name: GPTJEngine,
        LLamaEngine.config_name: LLamaEngine,
        GPT2Engine.config_name: GPT2Engine,
    }
