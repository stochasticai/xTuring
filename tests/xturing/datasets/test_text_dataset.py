import pytest
from datasets import Dataset

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


@pytest.mark.parametrize("data", [DATASET_EXAMPLE_DICT, DATASET_OTHER_EXAMPLE_DICT])
def test_save_and_reload(tmp_path, data):
    dataset = TextDataset(data)
    dataset.save(str(tmp_path))

    reloaded = TextDataset(str(tmp_path))
    assert len(reloaded) == 2
    assert list(reloaded) == list(dataset)


def test_hf_dataset_input():
    dataset = TextDataset(Dataset.from_dict(DATASET_OTHER_EXAMPLE_DICT))
    assert len(dataset) == 2
    assert dataset[0] == {"text": "first text", "target": "first text"}


def test_load_bare_dataset_directory(tmp_path):
    Dataset.from_dict(DATASET_EXAMPLE_DICT).save_to_disk(str(tmp_path))

    dataset = TextDataset(str(tmp_path))
    assert len(dataset) == 2
    assert dataset[1] == {"text": "second text"}
