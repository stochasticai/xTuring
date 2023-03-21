from turing.registry import BaseParent
from turing.trainers.lightning_trainer import LightningTrainer


class BaseTrainer(BaseParent):
    registry = {
        LightningTrainer.config_name: LightningTrainer,
    }
