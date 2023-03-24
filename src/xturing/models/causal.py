from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xturing.config import DEFAULT_DEVICE
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.datasets.text_dataset import TextDataset
from xturing.engines.base import BaseEngine
from xturing.models.base import BaseModel
from xturing.preprocessors.base import BasePreprocessor
from xturing.trainers.base import BaseTrainer
from xturing.trainers.lightning_trainer import LightningTrainer


class CausalModel(BaseModel):
    def __init__(self, engine: str, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create(engine, weights_path)

    def _make_collate_fn(self, dataset: Union[TextDataset, InstructionDataset]):
        return BasePreprocessor.create(
            dataset.config_name,
            self.engine.tokenizer,
            512,
            dataset.meta,
        )

    def _make_trainer(self, dataset: Union[TextDataset, InstructionDataset]):
        return BaseTrainer.create(
            LightningTrainer.config_name,
            self.engine,
            dataset,
            self._make_collate_fn(dataset),
        )

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        trainer = self._make_trainer(dataset)
        trainer.fit()

    def evaluate(self, dataset: Union[TextDataset, InstructionDataset]):
        pass

    def _generate_from_iterable(self, data_iterator: Iterable, do_tokenization=False):
        outputs = []

        for i, batch in enumerate(tqdm(data_iterator)):
            if do_tokenization:
                inputs = self.engine.tokenizer(batch, return_tensors="pt")
                input_ids = inputs.input_ids.to(DEFAULT_DEVICE)
            else:
                input_ids = batch["input_ids"].to(DEFAULT_DEVICE)
            with torch.no_grad():
                with torch.autocast("cuda"):
                    output = self.engine.model.generate(
                        input_ids=input_ids, do_sample=False, max_new_tokens=300
                    )

            output = self.engine.tokenizer.decode(output[0], skip_special_tokens=False)
            outputs.append(output)

        return outputs

    def generate(
        self,
        *,
        texts: Optional[Union[List[str], str]] = None,
        dataset: Optional[Union[TextDataset, InstructionDataset]] = None,
    ):
        self.engine.model.eval()
        self.engine.model = self.engine.model.to(DEFAULT_DEVICE)

        outputs = []

        if texts is not None:
            flattened_texts = [texts] if isinstance(texts, str) else texts

            outputs.extend(
                self._generate_from_iterable(flattened_texts, do_tokenization=True)
            )

        if dataset is not None:
            collate_fn = self._make_collate_fn(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

            outputs.extend(
                self._generate_from_iterable(dataloader, do_tokenization=False)
            )

        if texts is None and dataset is None:
            assert False, "Make sure texts or dataset is not None"

        if isinstance(texts, str) and dataset is None:
            return outputs[0]

        return outputs

    def save(self, path: Union[str, Path]):
        self.engine.save(path)


class CausalLoraModel(CausalModel):
    def __init__(self, engine: str, weights_path: Optional[str] = None):
        super().__init__(engine, weights_path)

    def _make_trainer(self, dataset: Union[TextDataset, InstructionDataset]):
        return BaseTrainer.create(
            LightningTrainer.config_name,
            self.engine,
            dataset,
            self._make_collate_fn(dataset),
            3,
            8,
            4e-3,
            "adamw",
            True,
            True,
        )
