import pytest
from datasets import Dataset as HFDataset
from datasets import load_from_disk

from xturing.datasets import InstructionDataset

DATASET_WRONG_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
}

DATASET_WRONG_DICT = {
    "data": ["first text", "second text"],
}

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
    "instruction": ["first instruction", "second instruction"],
}


def test_simple_dataset():
    with pytest.raises(AssertionError):
        dataset = InstructionDataset(DATASET_WRONG_EXAMPLE_DICT)


def test_dataset_additional_column():
    with pytest.raises(AssertionError):
        dataset = InstructionDataset(DATASET_WRONG_DICT)


def test_features_dataset():
    dataset = InstructionDataset(DATASET_OTHER_EXAMPLE_DICT)
    assert len(dataset) == 2
    assert dataset[0] == {
        "text": "first text",
        "target": "first text",
        "instruction": "first instruction",
    }
    assert dataset[1] == {
        "text": "second text",
        "target": "second text",
        "instruction": "second instruction",
    }
    dataset.save(".")
