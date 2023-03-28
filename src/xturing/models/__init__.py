from .base import BaseModel
from .distilgpt2 import DistilGPT2, DistilGPT2LORA
from .gpt2 import GPT2, GPT2LORA, GPT2Int8, GPT2LORAInt8
from .gptj import GPTJ, GPTJLORA, GPTJInt8, GPTJLORAInt8
from .llama import LLaMA, LLaMAInt8, LLaMALORA, LLaMALORAInt8
from .stable_diffusion import StableDiffusion

BaseModel.add_to_registry(DistilGPT2.config_name, DistilGPT2)
BaseModel.add_to_registry(DistilGPT2LORA.config_name, DistilGPT2LORA)
BaseModel.add_to_registry(GPTJ.config_name, GPTJ)
BaseModel.add_to_registry(GPTJInt8.config_name, GPTJInt8)
BaseModel.add_to_registry(GPTJLORAInt8.config_name, GPTJLORAInt8)
BaseModel.add_to_registry(LLaMA.config_name, LLaMA)
BaseModel.add_to_registry(StableDiffusion.config_name, StableDiffusion)
BaseModel.add_to_registry(GPT2.config_name, GPT2)
BaseModel.add_to_registry(GPTJLORA.config_name, GPTJLORA)
BaseModel.add_to_registry(LLaMALORA.config_name, LLaMALORA)
BaseModel.add_to_registry(GPT2LORA.config_name, GPT2LORA)
BaseModel.add_to_registry(GPT2Int8.config_name, GPT2Int8)
BaseModel.add_to_registry(GPT2LORAInt8.config_name, GPT2LORAInt8)
BaseModel.add_to_registry(LLaMAInt8.config_name, LLaMAInt8)
BaseModel.add_to_registry(LLaMALORAInt8.config_name, LLaMALORAInt8)
