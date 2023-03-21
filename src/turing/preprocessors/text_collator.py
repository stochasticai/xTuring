from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TextDataCollator:
    config_name = "text_dataset"

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, max_length: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batches: List[Dict]):
        # batches
        flatten_samples = []

        for sample in batches:
            input_text = self.tokenizer(sample["text"])
            input_ids = input_text["input_ids"]

            if "target" in sample:
                input_target = self.tokenizer(sample["target"])
                input_ids += input_target["input_ids"]

            input_ids = input_ids[: self.max_length - 1]
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask = [1] * len(input_ids)

            flatten_samples.append(
                {
                    "input_ids": torch.tensor(input_ids).long(),
                    "attention_mask": torch.tensor(attention_mask).long(),
                }
            )

        batch = self.tokenizer.pad(
            flatten_samples,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dim = batch["input_ids"].shape[-1]

        batch["targets"] = torch.roll(batch["input_ids"], -1, -1)

        return batch
