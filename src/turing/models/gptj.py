from pathlib import Path
from typing import List, Union, Optional
from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.preprocessors.base import BasePreprocessor
from turing.trainers.base import BaseTrainer


class GPTJ:
    def __init__(
            self, 
            weights_path: str,
            dataset_type: str,
            dataset_path: Union[str, Path]
        ):

        assert dataset_type in ["text_dataset", "instruction_dataset"], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        engine = BaseEngine("llama")(weights_path)
        dataset = BaseDataset(dataset_type)(dataset_path)
        collate_fn = BasePreprocessor(dataset_type)(engine.tokenizer, 512)
        
        self.trainer = BaseTrainer("lightning_trainer")(engine, dataset, collate_fn)
    
    def finetune(
        self,
        dataset: Union[TextDataset, InstructionDataset]
    ):
        self.trainer.fit()
         

    def evaluate(
        self,
        dataset: Union[TextDataset, InstructionDataset]
    ):
        pass

    def generate(
        self,
        texts: Optional[Union[List[str], str]] = None,
        dataset: Optional[Union[TextDataset, InstructionDataset]] = None
    ):
        pass

    def save(
        self,
        path: Union[str, Path]
    ):
        pass