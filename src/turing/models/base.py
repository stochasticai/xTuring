from turing.models.gpt2 import GPT2
from turing.models.gptj import GPTJ
from turing.models.llama import Llama
from turing.models.stable_diffusion import StableDiffusion
from turing.registry import BaseParent


class BaseModel(BaseParent):
    registry = {
        GPTJ.config_name: GPTJ,
        Llama.config_name: Llama,
        StableDiffusion.config_name: StableDiffusion,
    }
