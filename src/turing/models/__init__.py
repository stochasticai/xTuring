from .base import BaseModel
from .gpt2 import GPT2
from .gptj import GPTJ
from .llama import Llama
from .stable_diffusion import StableDiffusion

BaseModel.add_to_registry(GPTJ.config_name, GPTJ)
BaseModel.add_to_registry(Llama.config_name, Llama)
BaseModel.add_to_registry(StableDiffusion.config_name, StableDiffusion)
BaseModel.add_to_registry(GPT2.config_name, GPT2)
