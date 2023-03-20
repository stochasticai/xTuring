from registry import BaseParent

from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text2image_dataset import Text2ImageDataset
from turing.datasets.text_dataset import TextDataset


class BaseDataset(BaseParent):
    def __init__(self):
        super().__init__(
            registry={
                TextDataset.config_name: TextDataset,
                InstructionDataset.config_name: InstructionDataset,
                Text2ImageDataset.config_name: Text2ImageDataset,
            }
        )
