import pytest
from datasets import Dataset, DatasetDict

from xturing.datasets import PreferenceDataset

DATASET_DICT = {
    "prompt": ["What is AI?", "Explain gravity."],
    "chosen": ["AI is a field of computer science.", "Gravity is a force."],
    "rejected": ["AI is magic.", "Gravity is fake."],
}


def test_save_and_reload(tmp_path):
    dataset = PreferenceDataset(DATASET_DICT)
    dataset.save(str(tmp_path))

    reloaded = PreferenceDataset(str(tmp_path))
    assert len(reloaded) == 2
    assert list(reloaded) == list(dataset)


def test_hf_dataset_input():
    dataset = PreferenceDataset(Dataset.from_dict(DATASET_DICT))
    assert len(dataset) == 2
    assert dataset[0]["rejected"] == "AI is magic."


def test_load_dataset_dict_directory(tmp_path):
    DatasetDict({"train": Dataset.from_dict(DATASET_DICT)}).save_to_disk(str(tmp_path))

    dataset = PreferenceDataset(str(tmp_path))
    assert len(dataset) == 2
    assert dataset[1]["chosen"] == "Gravity is a force."


def test_hf_dataset_input_still_validates_columns():
    with pytest.raises(AssertionError):
        PreferenceDataset(Dataset.from_dict({"prompt": ["p"], "chosen": ["c"]}))
