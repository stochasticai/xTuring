from turing.models.stable_diffusion import StableDiffusion
from turing.preprocessors.instruction_collator import InstructionDataCollator
from turing.preprocessors.text_collator import TextDataCollator
from turing.registry import BaseParent


class BasePreprocessor(BaseParent):
    registry = {}


BasePreprocessor.add_to_registry(
    InstructionDataCollator.config_name, InstructionDataCollator
)
BasePreprocessor.add_to_registry(TextDataCollator.config_name, TextDataCollator)
