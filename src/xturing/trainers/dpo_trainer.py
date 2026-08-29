import copy
import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except Exception as import_err:  # pragma: no cover - optional dependency
    DeepSpeedCPUAdam = None
    _DEEPSPEED_IMPORT_ERROR = import_err
else:
    _DEEPSPEED_IMPORT_ERROR = None
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.trainer.trainer import Trainer

from xturing.config import DEFAULT_DEVICE, IS_INTERACTIVE
from xturing.datasets.base import BaseDataset
from xturing.engines.base import BaseEngine
from xturing.preprocessors.base import BasePreprocessor
from xturing.utils.logging import configure_logger

logger = configure_logger(__name__)


def compute_logprobs(model, input_ids, attention_mask, labels):
    """Compute per-token log-probabilities for the given labels.

    Tokens where ``labels == -100`` are ignored (prompt tokens). Returns the
    sum of log-probabilities over valid response tokens for each sample in the
    batch.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift so that logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Per-token log-probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_logprobs = torch.gather(
        log_probs, dim=2, index=shift_labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)

    # Mask out prompt tokens (labels == -100)
    loss_mask = shift_labels != -100
    per_token_logprobs = per_token_logprobs * loss_mask

    # Sum log-probs over response tokens for each sample
    return per_token_logprobs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logprobs,
    policy_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
):
    """Compute the DPO loss.

    Args:
        policy_chosen_logprobs: Log-probs of chosen responses under the policy.
        policy_rejected_logprobs: Log-probs of rejected responses under the policy.
        reference_chosen_logprobs: Log-probs of chosen responses under the
            frozen reference model.
        reference_rejected_logprobs: Log-probs of rejected responses under the
            frozen reference model.
        beta: Temperature parameter controlling deviation from the reference
            model.  Higher values penalise divergence more strongly.

    Returns:
        Scalar loss, chosen rewards, rejected rewards.
    """
    chosen_rewards = beta * (policy_chosen_logprobs - reference_chosen_logprobs)
    rejected_rewards = beta * (policy_rejected_logprobs - reference_rejected_logprobs)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


class DPOLightningModule(pl.LightningModule):
    """PyTorch Lightning module for DPO training."""

    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: Optional[BasePreprocessor] = None,
        batch_size: int = 2,
        learning_rate: float = 5e-7,
        optimizer_name: str = "adamw",
        beta: float = 0.1,
        saved_path: str = "saved_model",
    ):
        super().__init__()
        self.model_engine = model_engine
        self.pytorch_model = self.model_engine.model
        self.train_dataset = train_dataset
        self.preprocessor = preprocessor

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.beta = beta
        self.saved_path = saved_path

        self.losses = []

        # Create frozen reference model as a deep copy
        self.ref_model = copy.deepcopy(self.pytorch_model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "cpu_adam":
            if DeepSpeedCPUAdam is None:
                raise ModuleNotFoundError(
                    "DeepSpeed is required for optimizer 'cpu_adam'. "
                    "Install it with `pip install deepspeed`."
                ) from _DEEPSPEED_IMPORT_ERROR
            optimizer = DeepSpeedCPUAdam(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        self.train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.preprocessor,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            batch_size=self.batch_size,
        )
        return self.train_dl

    def training_step(self, batch, batch_idx):
        # Compute policy log-probabilities
        policy_chosen_logprobs = compute_logprobs(
            self.pytorch_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        policy_rejected_logprobs = compute_logprobs(
            self.pytorch_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

        # Compute reference log-probabilities (no gradients)
        with torch.no_grad():
            ref_chosen_logprobs = compute_logprobs(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_rejected_logprobs = compute_logprobs(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logprobs,
            policy_rejected_logprobs,
            ref_chosen_logprobs,
            ref_rejected_logprobs,
            beta=self.beta,
        )

        # Log metrics
        self.losses.append(loss.item())
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        self.log("loss", loss.item(), prog_bar=True)
        self.log("reward_margin", reward_margin, prog_bar=True)

        return loss

    def on_save_checkpoint(self, checkpoint):
        self.model_engine.save(self.saved_path)


class DPOTrainer:
    """Trainer for Direct Preference Optimization (DPO).

    Follows the same interface as :class:`LightningTrainer` so it can be
    used as a drop-in replacement via the trainer registry.
    """

    config_name: str = "dpo_trainer"

    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: BasePreprocessor,
        max_epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 5e-7,
        optimizer_name: str = "adamw",
        beta: float = 0.1,
        gradient_accumulation_steps: int = 1,
        logging_steps: int = 50,
        max_grad_norm: float = 2.0,
        save_total_limit: int = 4,
        output_dir: str = "saved_model",
        use_lora: bool = False,
        use_deepspeed: bool = False,
        deepspeed_config_path: Optional[str] = None,
        max_training_time_in_secs: Optional[int] = None,
        lora_type: int = 16,
        logger: Union[Logger, Iterable[Logger], bool] = True,
    ):
        self.lightning_model = DPOLightningModule(
            model_engine=model_engine,
            train_dataset=train_dataset,
            preprocessor=preprocessor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            beta=beta,
            saved_path=output_dir,
        )

        checkpoints_dir_path = Path(output_dir)

        if not checkpoints_dir_path.exists():
            checkpoints_dir_path.mkdir(exist_ok=True, parents=True)

        training_callbacks = []

        if max_training_time_in_secs is not None:
            training_callbacks.append(
                callbacks.Timer(
                    duration=datetime.timedelta(seconds=max_training_time_in_secs)
                )
            )

        model_engine.model.train()

        try:
            model_engine.model.print_trainable_parameters()
        except AttributeError:
            pass

        log_every_n_steps = max(1, int(logging_steps))
        accumulate_grad_batches = max(1, int(gradient_accumulation_steps))
        gradient_clip_val = max(0.0, float(max_grad_norm))

        if DEFAULT_DEVICE.type == "cpu":
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="cpu",
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=False,
                log_every_n_steps=log_every_n_steps,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                logger=logger,
            )
        elif not use_lora and not use_deepspeed:
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="gpu",
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=True,
                log_every_n_steps=log_every_n_steps,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                logger=logger,
            )
        else:
            training_callbacks = [
                callbacks.ModelCheckpoint(
                    dirpath=str(checkpoints_dir_path),
                    save_on_train_epoch_end=True,
                    save_top_k=max(1, int(save_total_limit)),
                ),
            ]

            strategy = "auto"
            if use_deepspeed:
                if DeepSpeedCPUAdam is None:
                    raise ModuleNotFoundError(
                        "use_deepspeed=True requires DeepSpeed. Install it with `pip install deepspeed`."
                    ) from _DEEPSPEED_IMPORT_ERROR
                if not IS_INTERACTIVE:
                    strategy = (
                        "deepspeed_stage_2_offload"
                        if optimizer_name == "cpu_adam"
                        else "deepspeed_stage_2"
                    )
                if deepspeed_config_path is not None:
                    strategy = DeepSpeedStrategy(config=deepspeed_config_path)
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="gpu",
                strategy=strategy,
                precision=lora_type,
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=True,
                log_every_n_steps=log_every_n_steps,
                accumulate_grad_batches=accumulate_grad_batches,
                gradient_clip_val=gradient_clip_val,
                logger=logger,
            )

    def fit(self):
        self.trainer.fit(self.lightning_model)
        if self.trainer.checkpoint_callback is not None:
            self.trainer.checkpoint_callback.best_model_path

    def engine(self):
        return self.lightning_model.model_engine
