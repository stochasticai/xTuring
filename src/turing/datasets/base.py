from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text2image_dataset import Text2ImageDataset
from turing.datasets.text_dataset import TextDataset
from turing.registry import BaseParent


class BaseDataset(BaseParent):
    registry = {}


BaseDataset.add_to_registry(TextDataset.config_name, TextDataset)
BaseDataset.add_to_registry(InstructionDataset.config_name, InstructionDataset)
BaseDataset.add_to_registry(Text2ImageDataset.config_name, Text2ImageDataset)
