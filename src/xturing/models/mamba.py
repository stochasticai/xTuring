from typing import Optional

from xturing.engines.mamba_engine import MambaEngine
from xturing.models.causal import CausalModel


class Mamba(CausalModel):
    config_name: str = "mamba"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MambaEngine.config_name, weights_path)
