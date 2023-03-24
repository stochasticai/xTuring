import datetime
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning import callbacks
from pytorch_lightning.trainer.trainer import Trainer

from xturing.config import DEFAULT_DEVICE
from xturing.datasets.base import BaseDataset
from xturing.engines.base import BaseEngine
from xturing.preprocessors.base import BasePreprocessor


class TuringLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: Optional[BasePreprocessor] = None,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        optimizer_name: str = "adamw",
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

        self.losses = []

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.adam(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "cpu_adam":
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
        loss = self.model_engine.training_step(batch)
        self.losses.append(loss.item())
        self.log("loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.model_engine.validation_step(batch)


class LightningTrainer:
    config_name: str = "lightning_trainer"

    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: BasePreprocessor,
        max_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adamw",
        use_lora: bool = False,
        use_deepspeed: bool = False,
        max_training_time_in_secs: Optional[int] = None,
    ):
        self.lightning_model = TuringLightningModule(
            model_engine=model_engine,
            train_dataset=train_dataset,
            preprocessor=preprocessor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
        )

        checkpoints_dir_path = Path("saved_model")

        if not checkpoints_dir_path.exists():
            checkpoints_dir_path.mkdir(exist_ok=True, parents=True)

        training_callbacks = [
            callbacks.LearningRateFinder(),
            callbacks.BatchSizeFinder(),
            # callbacks.ModelCheckpoint(
            #     dirpath=str(checkpoints_dir_path),
            #     save_top_k=3,
            #     monitor="loss",
            #     mode="min",  # Best model = min loss
            #     every_n_train_steps=200,
            # ),
        ]
        if max_training_time_in_secs is not None:
            training_callbacks.append(
                callbacks.Timer(
                    duration=datetime.timedelta(seconds=max_training_time_in_secs)
                )
            )

        if DEFAULT_DEVICE.type == "cpu":
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="cpu",
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=False,
                log_every_n_steps=50,
            )
        elif not use_lora and not use_deepspeed:
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="gpu",
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=True,
                log_every_n_steps=50,
            )
        else:
            training_callbacks = [
                callbacks.ModelCheckpoint(
                    dirpath=str(checkpoints_dir_path), save_on_train_epoch_end=True
                ),
            ]

            self.trainer = Trainer(
                num_nodes=1,
                accelerator="gpu",
                strategy="deepspeed_stage_2_offload"
                if optimizer_name == "cpu_adam"
                else "deepspeed_stage_2",
                precision=16,
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=True,
                log_every_n_steps=50,
            )

    def fit(self):
        self.trainer.fit(self.lightning_model)
        if self.trainer.checkpoint_callback is not None:
            self.trainer.checkpoint_callback.best_model_path

    def engine(self):
        return self.lightning_model.model_engine
