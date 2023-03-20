from registry import BaseParent
from turing.preprocessors.instruction_collator import InstructionDataCollator
from turing.preprocessors.text_collator import TextDataCollator
from turing.models.stable_diffusion import StableDiffusion

class BasePreprocessor(BaseParent):
    def __init__(self):
        super().__init__(registry={
            'instruction_dataset': InstructionDataCollator,
            'text_dataset': TextDataCollator
        })
