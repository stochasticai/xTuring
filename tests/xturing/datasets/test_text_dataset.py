import pytest
from datasets import Dataset as HFDataset
from datasets import load_from_disk

from xturing.datasets import TextDataset

DATASET_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
}

DATASET_WRONG_DICT = {
    "data": ["first text", "second text"],
}

DATASET_OTHER_EXAMPLE_DICT = {
    "text": ["first text", "second text"],
    "target": ["first text", "second text"],
}


def test_simple_dataset():
    dataset = TextDataset(DATASET_EXAMPLE_DICT)
    assert len(dataset) == 2
    assert dataset[0] == {"text": "first text"}
    assert dataset[1] == {"text": "second text"}


def test_dataset_additional_column():
    with pytest.raises(AssertionError):
        dataset = TextDataset(DATASET_WRONG_DICT)


def test_features_dataset():
    dataset = TextDataset(DATASET_OTHER_EXAMPLE_DICT)
    assert len(dataset) == 2
    assert dataset[0] == {"text": "first text", "target": "first text"}
    assert dataset[1] == {"text": "second text", "target": "second text"}
