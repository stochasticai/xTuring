from .base import BaseModel
from .gpt2 import GPT2, GPT2LORA
from .gptj import GPTJ, GPTJLORA
from .llama import Llama, LlamaLORA
from .stable_diffusion import StableDiffusion

BaseModel.add_to_registry(GPTJ.config_name, GPTJ)
BaseModel.add_to_registry(Llama.config_name, Llama)
BaseModel.add_to_registry(StableDiffusion.config_name, StableDiffusion)
BaseModel.add_to_registry(GPT2.config_name, GPT2)
BaseModel.add_to_registry(GPTJLORA.config_name, GPTJLORA)
BaseModel.add_to_registry(LlamaLORA.config_name, LlamaLORA)
BaseModel.add_to_registry(GPT2LORA.config_name, GPT2LORA)
