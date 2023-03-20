from registry import BaseParent

from .lightning import LightningTrainer


class BaseTrainer(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                "lightning_trainer": LightningTrainer,
            }
        )
