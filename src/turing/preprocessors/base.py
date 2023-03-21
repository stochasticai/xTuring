from registry import BaseParent

from turing.models.stable_diffusion import StableDiffusion
from turing.preprocessors.instruction_collator import InstructionDataCollator
from turing.preprocessors.text_collator import TextDataCollator


class BasePreprocessor(BaseParent):
    registry = {
        InstructionDataCollator.config_name: InstructionDataCollator,
        TextDataCollator.config_name: TextDataCollator,
    }
