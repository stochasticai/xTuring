from turing.engines.gpt2_engine import GPT2Engine, GPT2LoraEngine
from turing.engines.gptj_engine import GPTJEngine, GPTJLoraEngine
from turing.engines.llama_engine import LLamaEngine, LlamaLoraEngine
from turing.registry import BaseParent


class BaseEngine(BaseParent):
    registry = {
        GPTJEngine.config_name: GPTJEngine,
        LLamaEngine.config_name: LLamaEngine,
        GPT2Engine.config_name: GPT2Engine,
        GPTJLoraEngine.config_name: GPTJLoraEngine,
        LlamaLoraEngine.config_name: LlamaLoraEngine,
        GPT2LoraEngine.config_name: GPTJLoraEngine,
    }
