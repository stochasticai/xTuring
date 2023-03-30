from .base import BaseEngine
from .bloom_engine import (
    BloomEngine,
    BloomInt8Engine,
    BloomLoraEngine,
    BloomLoraInt8Engine,
)
from .cerebras_engine import (
    CerebrasEngine,
    CerebrasInt8Engine,
    CerebrasLoraEngine,
    CerebrasLoraInt8Engine,
)
from .distilgpt2_engine import DistilGPT2Engine, DistilGPT2LoraEngine
from .galactica_engine import (
    GalacticaEngine,
    GalacticaInt8Engine,
    GalacticaLoraEngine,
    GalacticaLoraInt8Engine,
)
from .gpt2_engine import GPT2Engine, GPT2Int8Engine, GPT2LoraEngine, GPT2LoraInt8Engine
from .gptj_engine import GPTJEngine, GPTJInt8Engine, GPTJLoraEngine, GPTJLoraInt8Engine
from .llama_engine import (
    LLamaEngine,
    LLamaInt8Engine,
    LlamaLoraEngine,
    LlamaLoraInt8Engine,
)
from .opt_engine import OPTEngine, OPTInt8Engine, OPTLoraEngine, OPTLoraInt8Engine

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
BaseEngine.add_to_registry(GalacticaEngine.config_name, GalacticaEngine)
BaseEngine.add_to_registry(GalacticaInt8Engine.config_name, GalacticaInt8Engine)
BaseEngine.add_to_registry(GalacticaLoraEngine.config_name, GalacticaLoraEngine)
BaseEngine.add_to_registry(GalacticaLoraInt8Engine.config_name, GalacticaLoraInt8Engine)
BaseEngine.add_to_registry(OPTEngine.config_name, OPTEngine)
BaseEngine.add_to_registry(OPTLoraEngine.config_name, OPTLoraEngine)
BaseEngine.add_to_registry(OPTInt8Engine.config_name, OPTInt8Engine)
BaseEngine.add_to_registry(OPTLoraInt8Engine.config_name, OPTLoraInt8Engine)
BaseEngine.add_to_registry(CerebrasEngine.config_name, CerebrasEngine)
BaseEngine.add_to_registry(CerebrasLoraEngine.config_name, CerebrasLoraEngine)
BaseEngine.add_to_registry(CerebrasInt8Engine.config_name, CerebrasInt8Engine)
BaseEngine.add_to_registry(CerebrasLoraInt8Engine.config_name, CerebrasLoraInt8Engine)
BaseEngine.add_to_registry(BloomEngine.config_name, BloomEngine)
BaseEngine.add_to_registry(BloomLoraEngine.config_name, BloomLoraEngine)
BaseEngine.add_to_registry(BloomInt8Engine.config_name, BloomInt8Engine)
BaseEngine.add_to_registry(BloomLoraInt8Engine.config_name, BloomLoraInt8Engine)
