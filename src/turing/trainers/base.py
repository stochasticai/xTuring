from turing.registry import BaseParent
from turing.trainers.lightning_trainer import LightningTrainer


class BaseTrainer(BaseParent):
    registry = {}


BaseTrainer.add_to_registry(LightningTrainer.config_name, LightningTrainer)
