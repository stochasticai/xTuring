from transformers import AutoModelForCausalLM
from pathlib import Path
from typing import List, Union, Optional
from turing.datasets.text_dataset import TextDataset
from turing.datasets.instruction_dataset import InstructionDataset

class Llama:
    def __init__(self, weights_path: str):
        pass
    
    def finetune(
        self,
        dataset: Union[TextDataset, InstructionDataset]
    ):
        pass

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