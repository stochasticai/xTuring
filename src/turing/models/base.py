from turing.models.gpt2 import GPT2
from turing.models.gptj import GPTJ
from turing.models.llama import Llama
from turing.models.stable_diffusion import StableDiffusion
from turing.registry import BaseParent


class BaseModel(BaseParent):
    registry = {}


BaseModel.add_to_registry(GPTJ.config_name, GPTJ)
BaseModel.add_to_registry(Llama.config_name, Llama)
BaseModel.add_to_registry(StableDiffusion.config_name, StableDiffusion)
BaseModel.add_to_registry(GPT2.config_name, GPT2)
