from turing.datasets import TextDataset
from turing.engines import GPT2LoraEngine
from turing.models import GPT2, BaseModel

EXAMPLE = "I want to be a part of the community"

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}

model = BaseModel.create("gpt2")


def test_text_gpt2():
    assert model.generate(texts="I want to")[: len(EXAMPLE)] == EXAMPLE


def test_text_dataset_gpt2():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    result = model.generate(dataset=dataset)
    assert len(result) == 2


def test_text_dataset_gpt2_lora():
    other_model = BaseModel.create("gpt2_lora")
    assert other_model.generate(texts="I want to")[: len(EXAMPLE)] == EXAMPLE
