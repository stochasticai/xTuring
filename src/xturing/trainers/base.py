from xturing.registry import BaseParent
from xturing.trainers.lightning_trainer import LightningTrainer


class BaseTrainer(BaseParent):
    registry = {}


BaseTrainer.add_to_registry(LightningTrainer.config_name, LightningTrainer)
