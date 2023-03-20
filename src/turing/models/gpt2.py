from pathlib import Path
from typing import List, Optional, Union

from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.gpt2_engine import GPT2Engine
from turing.preprocessors.text_collator import TextDataCollator
from turing.preprocessors.instruction_collator import InstructionDataCollator
from turing.trainers.lightning_trainer import LightningTrainer

from transformers import AutoTokenizer


class GPT2:
    def __init__(
        self, weights_path: Optional[str] = None
    ):
        self.engine = GPT2Engine(weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        if isinstance(dataset, TextDataCollator):
            collate_fn = TextDataCollator(tokenizer=self.tokenizer, max_length=512)
        else:
            collate_fn = InstructionDataCollator(tokenizer=self.tokenizer, max_length=512)

        self.trainer = LightningTrainer(self.engine, dataset, collate_fn)
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
