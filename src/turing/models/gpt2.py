from pathlib import Path
from typing import List, Optional, Union

from transformers import AutoTokenizer

from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.preprocessors.base import BasePreprocessor
from turing.preprocessors.instruction_collator import InstructionDataCollator
from turing.preprocessors.text_collator import TextDataCollator
from turing.trainers.base import BaseTrainer
from turing.trainers.lightning_trainer import LightningTrainer


class GPT2:
    def __init__(self, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create("gpt2_engine", weights_path)

        self.collate_fn = None
        self.trainer = None

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        self.collate_fn = BasePreprocessor.create(
            dataset.config_name, self.engine.tokenizer, 512
        )
        self.trainer = BaseTrainer.create(
            "lightning_trainer", self.engine, dataset, self.collate_fn
        )
        self.trainer.fit()

    def evaluate(self, dataset: Union[TextDataset, InstructionDataset]):
        pass

    def generate(
        self,
        texts: Optional[Union[List[str], str]] = None,
        dataset: Optional[Union[TextDataset, InstructionDataset]] = None,
    ):
        pass

    def save(self, path: Union[str, Path]):
        pass
