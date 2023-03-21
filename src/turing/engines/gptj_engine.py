from pathlib import Path
from typing import Optional, Union

import evaluate
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, GPTJForCausalLM

from turing.config import DEFAULT_DTYPE
from turing.utils.loss_fns import CrossEntropyLoss


class GPTJEngine:
    config_name: str = "gptj_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        if weights_path is None:
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.model = GPTJForCausalLM.from_pretrained(
                weights_path, torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

        self.loss_fct = CrossEntropyLoss()

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        if "label_mask" in batch:
            loss = self.loss_fct(
                outputs.get("logits"), batch["targets"], mask=batch["label_mask"]
            )
        else:
            loss = self.loss_fct(outputs.get("logits"), batch["targets"])

        return loss

    def validation_step(self, batch):
        metrics = evaluate.load("accuracy")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        logits = outputs.get("logits")
        preds = torch.argmax(logits, -1)
        acc = metrics.compute(preds, batch["targets"])

        return acc


class GPTJLoraEngine(GPTJEngine):
    config_name: str = "gptj_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(weights_path)
        if weights_path is None:
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.model = GPTJForCausalLM.from_pretrained(
                weights_path, torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        self.loss_fct = CrossEntropyLoss()
