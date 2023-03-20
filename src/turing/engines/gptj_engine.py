from typing import Union, Optional
from pathlib import Path
import torch
import torch.nn as nn
import evaluate
from transformers import GPTJForCausalLM

class GPTJEngine:
    def __init__(
        self, 
        weights_path: Optional[Union[str, Path]] = None
    ):
        if weights_path is None:
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            assert Path(weights_path).is_dir(), "The weights path should be a existing directory"
            self.model = GPTJForCausalLM.from_pretrained(weights_path)

        self.loss_fct = nn.CrossEntropyLoss()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        if "label_mask" in batch:
            loss = self.loss_fct(outputs.get("logits"), batch["targets"], mask=batch["label_mask"])
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