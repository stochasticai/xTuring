from typing import Optional

from xturing.engines.generic_engine import GenericEngine
from xturing.models.causal import CausalModel


class GenericModel(CausalModel):
    config_name: str = "generic"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GenericEngine.config_name, weights_path)
