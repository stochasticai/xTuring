from registry import BaseParent

from turing.models.gptj import GPTJ
from turing.models.llama import Llama
from turing.models.stable_diffusion import StableDiffusion


class BaseModel(BaseParent):
    registry = {
        GPTJ.config_name: GPTJ,
        Llama.config_name: Llama,
        StableDiffusion.config_name: StableDiffusion,
    }
