from xturing.datasets import InstructionDataset
from xturing.models import BaseModel

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
    "instruction": ["first instruction", "second instruction"],
}

DATASET_INFIX_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
    "instruction": ["{text} instruction: {target}", "{text} instruction: {target}"],
}

# text dataset tested in models/


def test_text_instruction():
    model = BaseModel.create("gpt2")
    dataset = InstructionDataset(DATASET_OTHER_EXAMPLE_DICT)
    result = model.generate(dataset=dataset)
    assert len(result) == 2


def test_text_instruction_infix():
    model = BaseModel.create("gpt2")
    dataset = InstructionDataset(DATASET_INFIX_EXAMPLE_DICT, infix_instruction=True)
    result = model.generate(dataset=dataset)
    assert len(result) == 2
