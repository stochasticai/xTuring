from registry import BaseParent

from turing.trainers.lightning_trainer import LightningTrainer


class BaseTrainer(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                LightningTrainer.config_name: LightningTrainer,
            }
        )
