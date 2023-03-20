from registry import BaseParent

from .gptj import GPTJ
from .llama import Llama


class BaseModel(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                "gptj": GPTJ,
                "llama": Llama,
                "stable_diffusion": StableDiffusion,
            }
        )
