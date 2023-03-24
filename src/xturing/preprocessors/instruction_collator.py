from typing import Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from xturing.datasets import InstructionDatasetMeta


class InstructionDataCollator:
    config_name = "instruction_dataset"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        meta: InstructionDatasetMeta = InstructionDatasetMeta(),
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = meta

    def _process_instruction(self, instruction, tags=None):
        # check if the instruction is valid
        # split the instruction into parts
        # check how many {text}/{target} parts are in the instruction
        if tags is None:
            tags = ["{text}", "{target}"]

        for tag in tags:
            assert (
                instruction.count(tag) == 1
            ), f"There should be exactly one {tag} in the instruction."

        parts = []

        for tag in tags:
            left, right = instruction.split(tag)
            parts.append(left)
            instruction = right

        parts.append(instruction)

        return parts

    def __call__(self, batches):
        flatten_samples = []
        label_masks = []

        for sample in batches:
            input_text = self.tokenizer(sample["text"])
            input_target = self.tokenizer(sample["target"])

            if self.meta.list_prompt_template is not None:
                combine = self.list_prompt_template.build(
                    instruction=sample["instruction"], text=sample["text"]
                )
                input_combine = self.tokenizer(combine)
                input_ids = input_combine["input_ids"] + input_target["input_ids"]
                label_mask = [False] * len(input_combine["input_ids"]) + [True] * len(
                    input_target["input_ids"]
                )
            elif not self.meta.infix_instruction:
                input_instruction = self.tokenizer(sample["instruction"])
                input_ids = (
                    input_instruction["input_ids"]
                    + input_text["input_ids"]
                    + input_target["input_ids"]
                )

                label_mask = (
                    [False] * len(input_instruction["input_ids"])
                    + [False] * len(input_text["input_ids"])
                    + [True] * len(input_target["input_ids"])
                )
            else:
                parts = self._process_instruction(sample["instruction"])

                input_instructions = [self.tokenizer(part) for part in parts]

                assert (
                    len(input_instructions) == 3
                ), "There should be exactly three parts in the instruction."

                input_ids = (
                    input_instructions[0]["input_ids"]
                    + input_text["input_ids"]
                    + input_instructions[1]["input_ids"]
                    + input_target["input_ids"]
                    + input_instructions[2]["input_ids"]
                )

                label_mask = (
                    [False] * len(input_instructions[0]["input_ids"])
                    + [False] * len(input_text["input_ids"])
                    + [False] * len(input_instructions[1]["input_ids"])
                    + [True] * len(input_target["input_ids"])
                    + [False] * len(input_instructions[2]["input_ids"])
                )

            input_ids = input_ids[: self.max_length - 1]
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask = [1] * len(input_ids)

            label_mask = label_mask[: self.max_length - 1]
            label_mask = label_mask + [True]

            flatten_samples.append(
                {
                    "input_ids": torch.tensor(input_ids).long(),
                    "attention_mask": torch.tensor(attention_mask).long(),
                }
            )
            label_masks.append(label_mask)

        batch = self.tokenizer.pad(
            flatten_samples,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dim = batch["input_ids"].shape[-1]

        batch["label_masks"] = torch.stack(
            [
                F.pad(torch.tensor(x), (0, dim - len(x)), value=False)
                for x in label_masks
            ]
        )
        batch["targets"] = torch.roll(batch["input_ids"], -1, -1)

        return batch
