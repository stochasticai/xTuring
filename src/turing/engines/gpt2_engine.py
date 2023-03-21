from pathlib import Path
from typing import Optional, Union

import evaluate
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from turing.config import DEFAULT_DTYPE


class GPT2Engine:
    config_name: str = "gpt2_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        if weights_path is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2", torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.model = AutoModelForCausalLM.from_pretrained(
                weights_path, torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.loss_fct = nn.CrossEntropyLoss()

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        if "label_mask" in batch:
            logits = outputs.get("logits").view(-1, outputs.get("logits").size(-1))
            targets = batch["targets"].view(-1)

            loss = self.loss_fct(logits, targets, mask=batch["label_mask"])
        else:
            logits = outputs.get("logits").view(-1, outputs.get("logits").size(-1))
            targets = batch["targets"].view(-1)
            loss = self.loss_fct(logits, targets)

        return loss

    def validation_step(self, batch):
        metrics = evaluate.load("accuracy")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        logits = outputs.get("logits")
        preds = torch.argmax(logits, -1)
        acc = metrics.compute(preds, batch["labels"])

        return acc


class GPT2LoraEngine(GPT2Engine):
    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(weights_path)
        if weights_path is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2", torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.model = AutoModelForCausalLM.from_pretrained(
                weights_path, torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)
        self.loss_fct = nn.CrossEntropyLoss()

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        self.loss_fct = nn.CrossEntropyLoss()
