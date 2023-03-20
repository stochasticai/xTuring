from turing.registry import BaseParent

from turing.models.gptj import GPTJ
from turing.models.llama import Llama
from turing.models.stable_diffusion import StableDiffusion
from turing.models.gpt2 import GPT2


class BaseModel(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                "gptj": GPTJ,
                "llama": Llama,
                "stable_diffusion": StableDiffusion,
                "gpt2": GPT2
            }
        )
