import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from xturing.datasets.base import BaseDataset


@dataclass
class PreferenceDatasetMeta:
    """Metadata for preference datasets used in DPO training."""


class PreferenceDataset(BaseDataset):
    """Dataset for Direct Preference Optimization (DPO) training.

    Each sample contains a prompt, a chosen (preferred) response, and a
    rejected (dispreferred) response. The dataset must have exactly three
    columns: ``prompt``, ``chosen``, and ``rejected``.

    Args:
        path: A local directory saved with ``datasets.save_to_disk``, a path
            to a ``.jsonl`` file, a HuggingFace ``Dataset``/``DatasetDict``,
            or a plain dictionary with the required keys.
    """

    config_name: str = "preference_dataset"

    def __init__(self, path: Union[str, Path, HFDataset, DatasetDict, dict]):
        if isinstance(path, HFDataset) or isinstance(path, DatasetDict):
            self.data = path
        elif isinstance(path, dict):
            self.data = {"train": HFDataset.from_dict(path)}
        else:
            path = Path(path)
            assert path.exists(), "path does not exist"
            if path.is_dir():
                self.data = load_from_disk(str(path))
            elif path.suffix == ".jsonl":
                self.data = {"train": HFDataset.from_dict(self._from_jsonl(path))}
            else:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. Use a directory or .jsonl file."
                )

        self._validate()
        self._meta = PreferenceDatasetMeta()

    def _from_jsonl(self, path: Path):
        data = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        try:
            for line in open(path):
                json_line = json.loads(line)
                data["prompt"].append(json_line["prompt"])
                data["chosen"].append(json_line["chosen"])
                data["rejected"].append(json_line["rejected"])
        except KeyError:
            raise ValueError(
                "The jsonl file should have keys: prompt, chosen, and rejected"
            )
        return data

    def _validate(self):
        assert "train" in self.data, "The dataset should have a train split"
        assert (
            "prompt" in self.data["train"].column_names
        ), "The dataset should have a column named prompt"
        assert (
            "chosen" in self.data["train"].column_names
        ), "The dataset should have a column named chosen"
        assert (
            "rejected" in self.data["train"].column_names
        ), "The dataset should have a column named rejected"
        assert (
            len(self.data["train"].column_names) == 3
        ), "The dataset should have only three columns: prompt, chosen, and rejected"

    def __len__(self):
        return len(self.data["train"])

    def __iter__(self):
        return iter(self.data["train"])

    def __getitem__(self, idx):
        return self.data["train"][idx]

    def save(self, path):
        return self.data["train"].save_to_disk(path)
