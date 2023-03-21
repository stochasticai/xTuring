import os
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.trainer.trainer import Trainer

from turing.datasets.base import BaseDataset
from turing.engines.base import BaseEngine
from turing.preprocessors.base import BasePreprocessor


class TuringLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: Optional[BasePreprocessor] = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
    ):
        super().__init__()
        self.model_engine = model_engine
        self.train_dataset = train_dataset
        self.preprocessor = preprocessor

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model_engine.model.parameters(), lr=self.learning_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        self.train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.preprocessor,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

        return self.train_dl

    def training_step(self, batch, batch_idx):
        return self.model_engine.training_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.model_engine.validation_step(batch)


class LightningTrainer:
    config_name = "lightning_trainer"

    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: BasePreprocessor,
        max_epochs: int = 3,
    ):
        self.lightning_model = TuringLightningModule(
            model_engine=model_engine,
            train_dataset=train_dataset,
            preprocessor=preprocessor,
        )

        training_callbacks = [
            callbacks.LearningRateFinder(),
            callbacks.BatchSizeFinder(),
        ]

        self.trainer = Trainer(
            num_nodes=1,
            accelerator="gpu",
            devices=torch.cuda.device_count(),
            max_epochs=max_epochs,
            callbacks=training_callbacks,
            enable_checkpointing=False,
            log_every_n_steps=50,
            precision=16,
        )

    def fit(self):
        self.trainer.fit(self.lightning_model)

    def engine(self):
        return self.lightning_model.model_engine
