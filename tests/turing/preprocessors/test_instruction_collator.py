from turing.datasets import InstructionDataset
from turing.models import BaseModel

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
    "instruction": ["first instruction", "second instruction"],
}


def test_text_instruction():
    model = BaseModel.create("gpt2")
    dataset = InstructionDataset(DATASET_OTHER_EXAMPLE_DICT)
    result = model.generate(dataset=dataset)
    assert len(result) == 2
