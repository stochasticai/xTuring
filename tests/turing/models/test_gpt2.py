from turing.datasets import TextDataset
from turing.models import GPT2

EXAMPLE = "I want to be a part of the community"

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}


def test_text_gpt2():
    model = GPT2()
    assert model.generate(texts="I want to")[: len(EXAMPLE)] == EXAMPLE


# def test_text_dataset_gpt2():
#     model = GPT2()
#     dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
#     print(model.generate(dataset=dataset))
