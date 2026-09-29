import pytest
from datasets import Dataset, DatasetDict

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


def test_save_and_reload(tmp_path):
    dataset = InstructionDataset(DATASET_OTHER_EXAMPLE_DICT)
    dataset.save(str(tmp_path))

    reloaded = InstructionDataset(str(tmp_path))
    assert len(reloaded) == 2
    assert list(reloaded) == list(dataset)


def test_hf_dataset_input():
    dataset = InstructionDataset(Dataset.from_dict(DATASET_OTHER_EXAMPLE_DICT))
    assert len(dataset) == 2
    assert dataset[1]["instruction"] == "second instruction"


def test_load_dataset_dict_directory(tmp_path):
    DatasetDict({"train": Dataset.from_dict(DATASET_OTHER_EXAMPLE_DICT)}).save_to_disk(
        str(tmp_path)
    )

    dataset = InstructionDataset(str(tmp_path))
    assert len(dataset) == 2
    assert dataset[0]["target"] == "first text"


def test_hf_dataset_input_still_validates_columns():
    with pytest.raises(AssertionError):
        InstructionDataset(Dataset.from_dict(DATASET_WRONG_EXAMPLE_DICT))
