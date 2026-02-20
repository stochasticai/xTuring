from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from xturing.datasets.preference_dataset import PreferenceDatasetMeta


class PreferenceDataCollator:
    """Collator for preference datasets used in DPO training.

    For each sample, this collator tokenizes two sequences:
    - ``prompt + chosen`` (the preferred completion)
    - ``prompt + rejected`` (the dispreferred completion)

    The resulting batch contains ``chosen_input_ids``, ``chosen_attention_mask``,
    ``chosen_labels``, and the corresponding ``rejected_*`` tensors. Labels are
    masked so that the loss is only computed over the response tokens (not the
    prompt).
    """

    config_name = "preference_dataset"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: Optional[int] = None,
        meta: PreferenceDatasetMeta = PreferenceDatasetMeta(),
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = meta

    def _tokenize_pair(self, prompt: str, response: str):
        """Tokenize a prompt-response pair and return input_ids with a label
        mask that marks only the response tokens as trainable."""
        prompt_tokens = self.tokenizer(prompt)
        response_tokens = self.tokenizer(response)

        input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
        # Labels: -100 for prompt tokens (ignored by loss), actual ids for response
        label_mask = [False] * len(prompt_tokens["input_ids"]) + [True] * len(
            response_tokens["input_ids"]
        )

        # Truncate to max_length - 1 to leave room for eos token
        input_ids = input_ids[: self.max_length - 1]
        input_ids.append(self.tokenizer.eos_token_id)
        attention_mask = [1] * len(input_ids)

        label_mask = label_mask[: self.max_length - 1]
        label_mask.append(True)

        return {
            "input_ids": torch.tensor(input_ids).long(),
            "attention_mask": torch.tensor(attention_mask).long(),
            "label_mask": label_mask,
        }

    def _pad_and_stack(self, samples: List[Dict]):
        """Pad a list of tokenized samples and stack into batch tensors."""
        padded = self.tokenizer.pad(
            [
                {"input_ids": s["input_ids"], "attention_mask": s["attention_mask"]}
                for s in samples
            ],
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        dim = padded["input_ids"].shape[-1]
        label_masks = torch.stack(
            [
                F.pad(
                    torch.tensor(s["label_mask"]),
                    (0, dim - len(s["label_mask"])),
                    value=False,
                )
                for s in samples
            ]
        )

        # Build labels: copy input_ids shifted by 1, masked with -100 for prompt tokens
        labels = padded["input_ids"].clone()
        labels[~label_masks] = -100

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
        }

    def __call__(self, batches: List[Dict]):
        chosen_samples = []
        rejected_samples = []

        for sample in batches:
            chosen_samples.append(
                self._tokenize_pair(sample["prompt"], sample["chosen"])
            )
            rejected_samples.append(
                self._tokenize_pair(sample["prompt"], sample["rejected"])
            )

        chosen_batch = self._pad_and_stack(chosen_samples)
        rejected_batch = self._pad_and_stack(rejected_samples)

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_batch["labels"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_batch["labels"],
        }
