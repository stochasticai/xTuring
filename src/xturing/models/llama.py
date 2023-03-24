from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xturing.datasets.base import BaseDataset
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.datasets.text_dataset import TextDataset
from xturing.engines.base import BaseEngine
from xturing.engines.llama_engine import LLamaEngine, LlamaLoraEngine
from xturing.models.causal import CausalLoraModel, CausalModel
from xturing.trainers.base import BaseTrainer
from xturing.trainers.lightning_trainer import LightningTrainer


class Llama(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLamaEngine.config_name, weights_path)

    def _make_trainer(self, dataset: Union[TextDataset, InstructionDataset]):
        return BaseTrainer.create(
            LightningTrainer.config_name,
            self.engine,
            dataset,
            self._make_collate_fn(dataset),
            3,
            1,
            1e-4,
            "cpu_adam",
            False,
            True,
        )


class LlamaLORA(CausalLoraModel):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraEngine.config_name, weights_path)

    def _make_trainer(self, dataset: Union[TextDataset, InstructionDataset]):
        return BaseTrainer.create(
            LightningTrainer.config_name,
            self.engine,
            dataset,
            self._make_collate_fn(dataset),
            3,
            4,
            1e-4,
            "adamw",
            True,
            True,
        )
