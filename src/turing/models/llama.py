from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.engines.llama_engine import LLamaEngine, LlamaLoraEngine
from turing.models.causal import CausalLoraModel, CausalModel
from turing.trainers.base import BaseTrainer
from turing.trainers.lightning_trainer import LightningTrainer


class Llama(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLamaEngine.config_name, weights_path)


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
            True,
            True,
        )
