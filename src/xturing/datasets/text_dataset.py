from dataclasses import dataclass
from pathlib import Path
from typing import Union

from datasets import Dataset as HFDataset
from datasets import load_from_disk

from xturing.datasets.base import BaseDataset


@dataclass
class TextDatasetMeta:
    pass


class TextDataset(BaseDataset):
    config_name: str = "text_dataset"

    def __init__(self, path: Union[str, Path, HFDataset, dict]):
        if isinstance(path, HFDataset):
            self.data = path
        elif isinstance(path, dict):
            self.data = {"train": HFDataset.from_dict(path)}
        else:
            assert Path(path).exists(), "path does not exist"
            self.data = load_from_disk(path)
        self._validate()
        self._meta = TextDatasetMeta()
        self._template = None

    def _validate(self):
        # check is hf dataset has train split and if it has column text, and if there are any other - it should be target
        assert "train" in self.data, "The dataset should have a train split"
        assert (
            "text" in self.data["train"].column_names
        ), "The dataset should have a column named text"

        if len(self.data["train"].column_names) > 1:
            assert (
                "target" in self.data["train"].column_names
            ), "The dataset should have a column named target if there is more than one column"
            assert (
                len(self.data["train"].column_names) == 2
            ), "The dataset should have only two columns, text and target"

    def __len__(self):
        return len(self.data["train"])

    def __iter__(self):
        return iter(self.data["train"])

    def __getitem__(self, idx):
        return self.data["train"][idx]
