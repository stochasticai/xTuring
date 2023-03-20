from registry import BaseParent

from turing.models.gptj import GPTJ
from turing.models.llama import Llama
from turing.models.stable_diffusion import StableDiffusion


class BaseModel(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                GPTJ.config_name: GPTJ,
                Llama.config_name: Llama,
                StableDiffusion.config_name: StableDiffusion,
            }
        )
