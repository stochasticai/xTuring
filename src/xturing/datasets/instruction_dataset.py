from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset as HFDataset
from datasets import load_from_disk

from xturing.datasets.base import BaseDataset


class ListPromptTemplate:
    def __init__(
        self,
        template: str,
        input_variables: List[str],
    ):
        self.template = template
        self.input_variables = input_variables

    def build(self, **kwargs) -> str:
        for i in self.input_variables:
            if i not in kwargs:
                raise ValueError(f"Missing input variable {i}")

        return self.template.format(**kwargs)


@dataclass
class InstructionDatasetMeta:
    infix_instruction: bool = False
    list_prompt_template: Optional[ListPromptTemplate] = None


class InstructionDataset(BaseDataset):
    config_name: str = "instruction_dataset"

    def __init__(
        self,
        path: Union[str, Path, HFDataset, dict],
        infix_instruction: bool = False,
        promt_template: str = None,
    ):
        if isinstance(path, HFDataset):
            self.data = path
        elif isinstance(path, dict):
            self.data = {"train": HFDataset.from_dict(path)}
        else:
            assert Path(path).exists(), "path does not exist"
            self.data = load_from_disk(path)
        self._validate()

        list_prompt_template = None

        if promt_template is not None:
            list_prompt_template = ListPromptTemplate(
                promt_template, input_variables=["instruction", "text"]
            )

        self._meta = InstructionDatasetMeta(
            infix_instruction=infix_instruction,
            list_prompt_template=list_prompt_template,
        )

    def _validate(self):
        # check is hf dataset has train split and if it has column text, and if there are any other - it should be target
        assert "train" in self.data, "The dataset should have a train split"
        assert (
            "text" in self.data["train"].column_names
        ), "The dataset should have a column named text"
        assert (
            "target" in self.data["train"].column_names
        ), "The dataset should have a column named target"
        assert (
            "instruction" in self.data["train"].column_names
        ), "The dataset should have a column named instruction"
        assert (
            len(self.data["train"].column_names) == 3
        ), "The dataset should have only three columns, instruction, text and target"

    def __len__(self):
        return len(self.data["train"])

    def __iter__(self):
        return iter(self.data["train"])

    def __getitem__(self, idx):
        return self.data["train"][idx]
