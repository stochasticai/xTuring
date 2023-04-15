import os
from pathlib import Path
from typing import Any, List, Optional, Union

import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xturing.config import DEFAULT_DEVICE, DEFAULT_DTYPE
from xturing.config.read_config import (
    exists_lora_config_file,
    exists_xturing_config_file,
)
from xturing.engines.base import BaseEngine
from xturing.engines.lora_engine import (
    LoraConfig,
    LoraModel,
    prepare_model_for_int8_training,
)
from xturing.utils.loss_fns import CrossEntropyLoss


class CausalEngine(BaseEngine):
    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        weights_path: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        load_8bit: Optional[bool] = False,
    ):
        self.model_name = model_name

        if weights_path is not None:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            if load_8bit:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
                self.model = AutoModelForCausalLM.from_pretrained(
                    weights_path,
                    torch_dtype=DEFAULT_DTYPE,
                    load_in_8bit=True,
                    device_map=device_map,
                )
                self.model = prepare_model_for_int8_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    weights_path, torch_dtype=DEFAULT_DTYPE
                )
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)
        elif model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_name is not None:
            if load_8bit:
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=DEFAULT_DTYPE,
                    load_in_8bit=True,
                    device_map=device_map,
                )
                for param in self.model.parameters():
                    param.data = param.data.contiguous()
                self.model = prepare_model_for_int8_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=DEFAULT_DTYPE
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(
                "Please provide a model_name, the weights path or model and tokenizer."
            )

        self.loss_fct = CrossEntropyLoss()
        self.load_8bit = load_8bit

    def training_step(self, batch):
        if self.load_8bit:
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                )
        else:
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
        acc = metrics.compute(preds, batch["labels"])

        return acc

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class CausalLoraEngine(CausalEngine):
    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        weights_path: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        load_8bit: Optional[bool] = False,
        target_modules: Optional[Union[List[str], str]] = None,
    ):
        # The base model should always be loaded from the original model
        # That's why weights_path is None. If not model.eval() will fail later
        super().__init__(
            model_name=model_name,
            weights_path=None
            if exists_xturing_config_file(weights_path)
            else weights_path,
            model=model,
            tokenizer=tokenizer,
            load_8bit=load_8bit,
        )

        # The model before applying LoRA
        self.base_model = self.model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            base_model_name_or_path=self.base_model.__dict__.get("name_or_path", None),
        )

        if len(target_modules) == 1:
            lora_config.fan_in_fan_out = True
            lora_config.enable_lora = [True, False, True]
        # self.model = LoraModel(lora_config, self.model)

        if weights_path is not None and exists_lora_config_file(weights_path):
            self.model = LoraModel.from_pretrained(self.base_model, weights_path)
        elif weights_path is not None and exists_xturing_config_file(weights_path):
            self.model = LoraModel(lora_config, self.model)
            model_weights_path = str(Path(weights_path).resolve() / "pytorch_model.bin")
            self.model.load_state_dict(
                torch.load(
                    model_weights_path  # , map_location=torch.device(DEFAULT_DEVICE)
                )
            )
        else:
            self.model = LoraModel(lora_config, self.model)
            self.model.print_trainable_parameters()

        self.loss_fct = CrossEntropyLoss()

    def save(self, saving_path: Union[str, Path]):
        # Save HF config file
        self.model.config.save_pretrained(str(saving_path))
        # Save model weights
        model_weights = str(Path(saving_path).resolve() / "pytorch_model.bin")

        torch.save(self.model.state_dict(), model_weights)
        # save adapter
        self.model.save_pretrained(saving_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(saving_path)
