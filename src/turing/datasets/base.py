from turing.registry import BaseParent

from .instruction_dataset import InstructionDataset
from .text2image_dataset import Text2ImageDataset
from .text_dataset import TextDataset


class BaseDataset(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                "text_dataset": TextDataset,
                "instruction_dataset": InstructionDataset,
                "text2image_dataset": Text2ImageDataset,
            }
        )
