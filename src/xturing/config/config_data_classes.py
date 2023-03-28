from typing import Optional

from pydantic import BaseModel, validator


class FinetuningConfig(BaseModel):
    learning_rate: float
    gradient_accumulation_steps: int
    batch_size: int
    weight_decay: float
    warmup_steps: int
    eval_steps: int
    save_steps: int
    max_length: int
    num_train_epochs: int
    logging_steps: int
    max_grad_norm: float
    save_total_limit: int
    optimizer_name: str
    output_dir: str

    @validator("optimizer_name")
    def validate_optimizer_name(cls, v):
        valid_optimizers = ["adamw", "adam", "cpu_adam"]

        assert v in valid_optimizers, f"{v} is not a valid optimizer for finetuning"

        return v


class GenerationConfig(BaseModel):
    penalty_alpha: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
