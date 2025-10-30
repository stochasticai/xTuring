from xturing.engines.base import BaseEngine
from xturing.engines.bloom_engine import (
    BloomEngine,
    BloomInt8Engine,
    BloomLoraEngine,
    BloomLoraInt8Engine,
)
from xturing.engines.cerebras_engine import (
    CerebrasEngine,
    CerebrasInt8Engine,
    CerebrasLoraEngine,
    CerebrasLoraInt8Engine,
)
from xturing.engines.distilgpt2_engine import DistilGPT2Engine, DistilGPT2LoraEngine
from xturing.engines.falcon_engine import (
    FalconEngine,
    FalconInt8Engine,
    FalconLoraEngine,
    FalconLoraInt8Engine,
    FalconLoraKbitEngine,
)
from xturing.engines.galactica_engine import (
    GalacticaEngine,
    GalacticaInt8Engine,
    GalacticaLoraEngine,
    GalacticaLoraInt8Engine,
)
from xturing.engines.generic_engine import (
    GenericEngine,
    GenericInt8Engine,
    GenericLoraEngine,
    GenericLoraInt8Engine,
    GenericLoraKbitEngine,
)
from xturing.engines.gpt2_engine import (
    GPT2Engine,
    GPT2Int8Engine,
    GPT2LoraEngine,
    GPT2LoraInt8Engine,
)
from xturing.engines.gpt_oss_engine import (
    GPTOSS20BEngine,
    GPTOSS20BInt8Engine,
    GPTOSS20BLoraEngine,
    GPTOSS20BLoraInt8Engine,
    GPTOSS20BLoraKbitEngine,
    GPTOSS120BEngine,
    GPTOSS120BInt8Engine,
    GPTOSS120BLoraEngine,
    GPTOSS120BLoraInt8Engine,
    GPTOSS120BLoraKbitEngine,
)
from xturing.engines.gptj_engine import (
    GPTJEngine,
    GPTJInt8Engine,
    GPTJLoraEngine,
    GPTJLoraInt8Engine,
)
from xturing.engines.llama2_engine import (
    LLama2Engine,
    LLama2Int8Engine,
    LLama2LoraEngine,
    LLama2LoraInt8Engine,
    LLama2LoraKbitEngine,
)
from xturing.engines.llama_engine import (
    LLamaEngine,
    LLamaInt8Engine,
    LlamaLoraEngine,
    LlamaLoraInt8Engine,
    LlamaLoraKbitEngine,
)
from xturing.engines.mamba_engine import MambaEngine
from xturing.engines.minimax_m2_engine import (
    MiniMaxM2Engine,
    MiniMaxM2Int8Engine,
    MiniMaxM2LoraEngine,
    MiniMaxM2LoraInt8Engine,
    MiniMaxM2LoraKbitEngine,
)
from xturing.engines.opt_engine import (
    OPTEngine,
    OPTInt8Engine,
    OPTLoraEngine,
    OPTLoraInt8Engine,
)
from xturing.engines.qwen_engine import (
    Qwen3Engine,
    Qwen3Int8Engine,
    Qwen3LoraEngine,
    Qwen3LoraInt8Engine,
    Qwen3LoraKbitEngine,
)

BaseEngine.add_to_registry(BloomEngine.config_name, BloomEngine)
BaseEngine.add_to_registry(BloomInt8Engine.config_name, BloomInt8Engine)
BaseEngine.add_to_registry(BloomLoraEngine.config_name, BloomLoraEngine)
BaseEngine.add_to_registry(BloomLoraInt8Engine.config_name, BloomLoraInt8Engine)
BaseEngine.add_to_registry(CerebrasEngine.config_name, CerebrasEngine)
BaseEngine.add_to_registry(CerebrasInt8Engine.config_name, CerebrasInt8Engine)
BaseEngine.add_to_registry(CerebrasLoraEngine.config_name, CerebrasLoraEngine)
BaseEngine.add_to_registry(CerebrasLoraInt8Engine.config_name, CerebrasLoraInt8Engine)
BaseEngine.add_to_registry(DistilGPT2Engine.config_name, DistilGPT2Engine)
BaseEngine.add_to_registry(DistilGPT2LoraEngine.config_name, DistilGPT2LoraEngine)
BaseEngine.add_to_registry(FalconEngine.config_name, FalconEngine)
BaseEngine.add_to_registry(FalconInt8Engine.config_name, FalconInt8Engine)
BaseEngine.add_to_registry(FalconLoraEngine.config_name, FalconLoraEngine)
BaseEngine.add_to_registry(FalconLoraInt8Engine.config_name, FalconLoraInt8Engine)
BaseEngine.add_to_registry(FalconLoraKbitEngine.config_name, FalconLoraKbitEngine)
BaseEngine.add_to_registry(GalacticaEngine.config_name, GalacticaEngine)
BaseEngine.add_to_registry(GalacticaInt8Engine.config_name, GalacticaInt8Engine)
BaseEngine.add_to_registry(GalacticaLoraEngine.config_name, GalacticaLoraEngine)
BaseEngine.add_to_registry(GalacticaLoraInt8Engine.config_name, GalacticaLoraInt8Engine)
BaseEngine.add_to_registry(GenericEngine.config_name, GenericEngine)
BaseEngine.add_to_registry(GenericInt8Engine.config_name, GenericInt8Engine)
BaseEngine.add_to_registry(GenericLoraEngine.config_name, GenericLoraEngine)
BaseEngine.add_to_registry(GenericLoraInt8Engine.config_name, GenericLoraInt8Engine)
BaseEngine.add_to_registry(GenericLoraKbitEngine.config_name, GenericLoraKbitEngine)
BaseEngine.add_to_registry(GPTJEngine.config_name, GPTJEngine)
BaseEngine.add_to_registry(GPTJInt8Engine.config_name, GPTJInt8Engine)
BaseEngine.add_to_registry(GPTJLoraEngine.config_name, GPTJLoraEngine)
BaseEngine.add_to_registry(GPTJLoraInt8Engine.config_name, GPTJLoraInt8Engine)
BaseEngine.add_to_registry(GPT2Engine.config_name, GPT2Engine)
BaseEngine.add_to_registry(GPT2Int8Engine.config_name, GPT2Int8Engine)
BaseEngine.add_to_registry(GPT2LoraEngine.config_name, GPT2LoraEngine)
BaseEngine.add_to_registry(GPT2LoraInt8Engine.config_name, GPT2LoraInt8Engine)
BaseEngine.add_to_registry(GPTOSS120BEngine.config_name, GPTOSS120BEngine)
BaseEngine.add_to_registry(GPTOSS120BInt8Engine.config_name, GPTOSS120BInt8Engine)
BaseEngine.add_to_registry(GPTOSS120BLoraEngine.config_name, GPTOSS120BLoraEngine)
BaseEngine.add_to_registry(
    GPTOSS120BLoraInt8Engine.config_name, GPTOSS120BLoraInt8Engine
)
BaseEngine.add_to_registry(
    GPTOSS120BLoraKbitEngine.config_name, GPTOSS120BLoraKbitEngine
)
BaseEngine.add_to_registry(GPTOSS20BEngine.config_name, GPTOSS20BEngine)
BaseEngine.add_to_registry(GPTOSS20BInt8Engine.config_name, GPTOSS20BInt8Engine)
BaseEngine.add_to_registry(GPTOSS20BLoraEngine.config_name, GPTOSS20BLoraEngine)
BaseEngine.add_to_registry(GPTOSS20BLoraInt8Engine.config_name, GPTOSS20BLoraInt8Engine)
BaseEngine.add_to_registry(GPTOSS20BLoraKbitEngine.config_name, GPTOSS20BLoraKbitEngine)
BaseEngine.add_to_registry(LLamaEngine.config_name, LLamaEngine)
BaseEngine.add_to_registry(LLamaInt8Engine.config_name, LLamaInt8Engine)
BaseEngine.add_to_registry(LlamaLoraEngine.config_name, LlamaLoraEngine)
BaseEngine.add_to_registry(LlamaLoraInt8Engine.config_name, LlamaLoraInt8Engine)
BaseEngine.add_to_registry(LlamaLoraKbitEngine.config_name, LlamaLoraKbitEngine)
BaseEngine.add_to_registry(LLama2Engine.config_name, LLama2Engine)
BaseEngine.add_to_registry(LLama2Int8Engine.config_name, LLama2Int8Engine)
BaseEngine.add_to_registry(LLama2LoraEngine.config_name, LLama2LoraEngine)
BaseEngine.add_to_registry(LLama2LoraInt8Engine.config_name, LLama2LoraInt8Engine)
BaseEngine.add_to_registry(LLama2LoraKbitEngine.config_name, LLama2LoraKbitEngine)
BaseEngine.add_to_registry(MambaEngine.config_name, MambaEngine)
BaseEngine.add_to_registry(MiniMaxM2Engine.config_name, MiniMaxM2Engine)
BaseEngine.add_to_registry(MiniMaxM2Int8Engine.config_name, MiniMaxM2Int8Engine)
BaseEngine.add_to_registry(MiniMaxM2LoraEngine.config_name, MiniMaxM2LoraEngine)
BaseEngine.add_to_registry(MiniMaxM2LoraInt8Engine.config_name, MiniMaxM2LoraInt8Engine)
BaseEngine.add_to_registry(MiniMaxM2LoraKbitEngine.config_name, MiniMaxM2LoraKbitEngine)
BaseEngine.add_to_registry(Qwen3Engine.config_name, Qwen3Engine)
BaseEngine.add_to_registry(Qwen3Int8Engine.config_name, Qwen3Int8Engine)
BaseEngine.add_to_registry(Qwen3LoraEngine.config_name, Qwen3LoraEngine)
BaseEngine.add_to_registry(Qwen3LoraInt8Engine.config_name, Qwen3LoraInt8Engine)
BaseEngine.add_to_registry(Qwen3LoraKbitEngine.config_name, Qwen3LoraKbitEngine)
BaseEngine.add_to_registry(OPTEngine.config_name, OPTEngine)
BaseEngine.add_to_registry(OPTInt8Engine.config_name, OPTInt8Engine)
BaseEngine.add_to_registry(OPTLoraEngine.config_name, OPTLoraEngine)
BaseEngine.add_to_registry(OPTLoraInt8Engine.config_name, OPTLoraInt8Engine)
