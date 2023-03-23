from xturing.datasets import TextDataset
from xturing.engines import GPT2LoraEngine
from xturing.models import GPT2, BaseModel

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


def test_train_gpt2():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    model = BaseModel.create("gpt2")
    model.finetune(dataset=dataset)
    result = model.generate(dataset=dataset)
    assert len(result) == 2


def test_train_gpt2_lora():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    model = BaseModel.create("gpt2_lora")
    model.finetune(dataset=dataset)
    result = model.generate(dataset=dataset)
    assert len(result) == 2
