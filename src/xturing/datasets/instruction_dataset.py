from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from datasets import Dataset as HFDataset
from datasets import load_from_disk

from xturing.datasets.base import BaseDataset


@dataclass
class InstructionDatasetMeta:
    infix_instruction: bool = False


class ListPromptTemplate:
    def __init__(
        self,
        template: str,
        input_variables: List[str],
        list_templates: Dict[str, str] = None,
    ):
        self.template = template
        self.input_variables = input_variables

        self.list_templates = (
            list_templates  # key words in the list template are number and text
        )
        if self.list_templates is None:
            self.list_templates = {}

    def check_list_template(self, list_template: str):
        return list_template in self.list_templates

    @classmethod
    def process_list_template(cls, inputs: List[str], list_template: str):
        return "\n".join(
            list_template.format(number=i, text=text) for i, text in enumerate(inputs)
        )

    def build(self, **kwargs) -> str:
        for i in self.input_variables:
            if i not in kwargs:
                raise ValueError(f"Missing input variable {i}")

        for k, v in kwargs.items():
            if isinstance(v, list):
                if k not in self.list_templates:
                    raise ValueError(f"Missing list template for variable {k}")
                kwargs[k] = self.process_list_template(v, self.list_templates[k])

        return self.template.format(**kwargs)


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
        self._meta = InstructionDatasetMeta(infix_instruction=infix_instruction)
        self._template = (
            ListPromptTemplate(promt_template, input_variables=["instruction", "text"])
            if promt_template != None
            else None
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
