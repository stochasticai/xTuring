from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.engines.llama_engine import LLamaEngine
from turing.models.causal import CausalModel


class Llama(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: str):
        super().__init__(LLamaEngine.config_name, weights_path)


class LlamaLORA(Llama):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create("llama_lora_engine", weights_path)

        self.collate_fn = None
        self.trainer = None

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        self.collate_fn = BasePreprocessor.create(
            dataset.config_name,
            self.engine.tokenizer,
            512,
        )
        self.trainer = BaseTrainer.create(
            "lightning_trainer",
            self.engine,
            dataset,
            self.collate_fn,
            3,
            8,
            4e-3,
            True,
            True,
        )
        self.trainer.fit()
