from registry import BaseParent
from turing.datasets.text_dataset import TextDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text2image_dataset import Text2ImageDataset


class BaseDataset(BaseParent):
    def __init__(self):
        super().__init__(registry={
            'text_dataset': TextDataset,
            'instruction_dataset': InstructionDataset,
            'text2image_dataset': Text2ImageDataset,
        })
