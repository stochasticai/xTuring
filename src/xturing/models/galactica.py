from typing import Optional

from xturing.engines.galactica_engine import (
    GalacticaEngine,
    GalacticaInt8Engine,
    GalacticaLoraEngine,
    GalacticaLoraInt8Engine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraModel,
    CausalModel,
)


class Galactica(CausalModel):
    config_name: str = "galactica"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GalacticaEngine.config_name, weights_path)


class GalacticaLora(CausalLoraModel):
    config_name: str = "galactica_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GalacticaLoraEngine.config_name, weights_path)


class GalacticaInt8(CausalInt8Model):
    config_name: str = "galactica_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GalacticaInt8Engine.config_name, weights_path)


class GalacticaLoraInt8(CausalLoraInt8Model):
    config_name: str = "galactica_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GalacticaLoraInt8Engine.config_name, weights_path)
