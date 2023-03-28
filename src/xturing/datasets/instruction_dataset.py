import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset as HFDataset
from datasets import load_from_disk

from xturing.datasets.base import BaseDataset
from xturing.model_apis.openai import DAVINCI
from xturing.self_instruct import (
    bootstrap_instructions,
    generate_instances,
    identify_if_classification,
    prepare_for_finetuning,
)
from xturing.utils.utils import create_temp_directory, no_std_out


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
            path = Path(path)
            assert Path(path).exists(), "path does not exist"

            if path.is_dir():
                self.data = load_from_disk(str(path))
            elif path.suffix == ".jsonl":
                self.data = {"train": HFDataset.from_dict(self.from_jsonl(path))}

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

    def from_jsonl(self, path: Path):
        data = {
            "text": [],
            "instruction": [],
            "target": [],
        }
        try:
            for line in open(path):
                json_line = json.loads(line)
                data["text"].append(json_line["text"])
                data["instruction"].append(json_line["instruction"])
                data["target"].append(json_line["target"])
        except KeyError:
            raise ValueError(
                "The jsonl file should have keys text, instruction and target"
            )
        return data

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

    @classmethod
    def generate_dataset(
        cls,
        api_key: str,
        path: str,
        organization: Optional[str] = None,
        engine: str = DAVINCI,
        num_instructions: int = 10,
        num_instructions_for_finetuning: int = 5,
        num_prompt_instructions: int = 1,
        request_batch_size: int = 1,
    ):
        cache_directory = create_temp_directory(
            f"./self_instruct_{engine}_cache_{num_instructions}_{num_instructions_for_finetuning}"
        )
        seed_tasks_path = Path(path)

        machine_generated = (
            Path(cache_directory) / "machine_generated_instructions.jsonl"
        )
        filtered = Path(cache_directory) / "filtered_instructions.jsonl"
        is_clf = Path(cache_directory) / "is_clf_or_not.jsonl"
        all_generated = Path(cache_directory) / "all_generated.jsonl"
        sampled_generated = Path(cache_directory) / "sampled_generated.jsonl"
        finetuning = Path(cache_directory) / "finetuning.jsonl"

        bootstrap_instructions.bootstrap_instructions(
            seed_tasks_path=seed_tasks_path,
            output_file=machine_generated,
            num_instructions_to_generate=num_instructions_for_finetuning,
            use_clf_seed_tasks_only=False,
            engine=engine,
            num_prompt_instructions=num_prompt_instructions,
            request_batch_size=request_batch_size,
            api_key=api_key,
            organization=organization,
        )

        identify_if_classification.identify_if_classification(
            input_file=machine_generated,
            output_file=is_clf,
            num_instructions=num_instructions,
            template="template_1",
            engine=engine,
            request_batch_size=request_batch_size,
            api_key=api_key,
            organization=organization,
        )

        generate_instances.generate_instances(
            input_file=machine_generated,
            classification_file=is_clf,
            output_file=filtered,
            num_instructions=num_instructions,
            max_instances_to_generate=num_instructions,
            generation_tasks_only=False,
            classification_tasks_only=False,
            engine=engine,
            request_batch_size=request_batch_size,
            api_key=api_key,
            organization=organization,
        )

        prepare_for_finetuning.prepare_for_finetuning(
            instance_files=[filtered],
            classification_type_files=[is_clf],
            all_generated=all_generated,
            sampled_generated=sampled_generated,
            finetuning=finetuning,
            num_instructions=num_instructions_for_finetuning,
            include_seed_tasks=True,
            seed_tasks_path=seed_tasks_path,
        )

        path = Path("./self_instruct_davinci_cache_10_5/sampled_generated.jsonl")
        return InstructionDataset(path)
