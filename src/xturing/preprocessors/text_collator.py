from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from xturing.datasets import TextDatasetMeta


class TextDataCollator:
    config_name = "text_dataset"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        meta: TextDatasetMeta = TextDatasetMeta(),
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = meta

    def __call__(self, batches: List[Dict]):
        # batches
        flatten_samples = []
        label_masks = []

        for sample in batches:
            input_text = self.tokenizer(sample["text"])
            input_ids = input_text["input_ids"]

            if "target" in sample:
                input_target = self.tokenizer(sample["target"])
                input_ids += input_target["input_ids"]

            input_ids = input_ids[: self.max_length - 1]
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask = [1] * len(input_ids)

            if "target" in sample:
                label_mask = [False] * len(input_text["input_ids"]) + [True] * len(
                    input_target["input_ids"]
                )
            else:
                label_mask = [True] * len(input_text["input_ids"])

            label_mask = label_mask[: self.max_length]
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
