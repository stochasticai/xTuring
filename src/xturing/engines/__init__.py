from .base import BaseEngine
from .gpt2_engine import GPT2Engine, GPT2LoraEngine
from .gptj_engine import GPTJEngine, GPTJLoraEngine
from .llama_engine import LLamaEngine, LlamaLoraEngine

BaseEngine.add_to_registry(GPTJEngine.config_name, GPTJEngine)
BaseEngine.add_to_registry(GPTJLoraEngine.config_name, GPTJLoraEngine)
BaseEngine.add_to_registry(LLamaEngine.config_name, LLamaEngine)
BaseEngine.add_to_registry(GPT2Engine.config_name, GPT2Engine)
BaseEngine.add_to_registry(GPT2LoraEngine.config_name, GPT2LoraEngine)
BaseEngine.add_to_registry(LlamaLoraEngine.config_name, LlamaLoraEngine)
