from .base import BaseEngine
from .distilgpt2_engine import DistilGPT2Engine, DistilGPT2LoraEngine
from .gpt2_engine import GPT2Engine, GPT2Int8Engine, GPT2LoraEngine, GPT2LoraInt8Engine
from .gptj_engine import GPTJEngine, GPTJInt8Engine, GPTJLoraEngine, GPTJLoraInt8Engine
from .llama_engine import (
    LLamaEngine,
    LLamaInt8Engine,
    LlamaLoraEngine,
    LlamaLoraInt8Engine,
)

BaseEngine.add_to_registry(DistilGPT2Engine.config_name, DistilGPT2Engine)
BaseEngine.add_to_registry(DistilGPT2LoraEngine.config_name, DistilGPT2LoraEngine)
BaseEngine.add_to_registry(GPTJEngine.config_name, GPTJEngine)
BaseEngine.add_to_registry(GPTJLoraEngine.config_name, GPTJLoraEngine)
BaseEngine.add_to_registry(GPTJInt8Engine.config_name, GPTJInt8Engine)
BaseEngine.add_to_registry(GPTJLoraInt8Engine.config_name, GPTJLoraInt8Engine)
BaseEngine.add_to_registry(GPT2Engine.config_name, GPT2Engine)
BaseEngine.add_to_registry(GPT2LoraEngine.config_name, GPT2LoraEngine)
BaseEngine.add_to_registry(GPT2Int8Engine.config_name, GPT2Int8Engine)
BaseEngine.add_to_registry(GPT2LoraInt8Engine.config_name, GPT2LoraInt8Engine)
BaseEngine.add_to_registry(LLamaEngine.config_name, LLamaEngine)
BaseEngine.add_to_registry(LlamaLoraEngine.config_name, LlamaLoraEngine)
BaseEngine.add_to_registry(LLamaInt8Engine.config_name, LLamaInt8Engine)
BaseEngine.add_to_registry(LlamaLoraInt8Engine.config_name, LlamaLoraInt8Engine)
