from typing import Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def gen_instruction_prompt(instruction, input):
    if len(input) > 0:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a comprehensive and informative response that appropriately completes the request. The response must have at least 50 words. Must not repeat text.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a comprehensive and informative response that appropriately completes the request. The response must have at least 50 words. Must not repeat text.

### Instruction:
{instruction}

### Response:
"""
    return prompt


class InstructionDataCollator:
    config_name = "instruction_dataset"

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, max_length: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batches):
        flatten_samples = []
        label_masks = []

        for sample in batches:
            prompt = gen_instruction_prompt(sample["instruction"], sample["text"])
            input_prompt = self.tokenizer(prompt)
            input_target = self.tokenizer(sample["target"])

            input_ids = input_prompt["input_ids"] + input_target["input_ids"]

            input_ids = input_ids[: self.max_length - 1]
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask = [1] * len(input_ids)

            label_mask = [False] * len(input_prompt["input_ids"]) + [True] * len(
                input_target["input_ids"]
            )
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
