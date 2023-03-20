from registry import BaseParent
from turing.models.gptj import GPTJ
from turing.models.llama import Llama
from turing.models.stable_diffusion import StableDiffusion

class BaseModel(BaseParent):
    def __init__(self):
        super().__init__(registry={
            'gptj': GPTJ,
            'llama': Llama,
            'stable_diffusion': StableDiffusion,
        })
