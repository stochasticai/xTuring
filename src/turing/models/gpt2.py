from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from turing.config import DEFAULT_DEVICE
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.preprocessors.base import BasePreprocessor
from turing.trainers.base import BaseTrainer


class GPT2:
    config_name: str = "gpt2"

    def __init__(self, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create("gpt2_engine", weights_path)

    def _make_collate_fn(self, dataset: Union[TextDataset, InstructionDataset]):
        return BasePreprocessor.create(dataset.config_name, self.engine.tokenizer, 512)

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        collate_fn = self._make_collate_fn(dataset)
        trainer = BaseTrainer.create(
            "lightning_trainer", self.engine, dataset, collate_fn
        )
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


class GPT2LORA(GPT2):
    config_name: str = "gpt2_lora"

    def __init__(self, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create("gpt2_lora_engine", weights_path)

        self.collate_fn = None
        self.trainer = None

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        self.collate_fn = BasePreprocessor.create(
            dataset.config_name,
            self.engine.tokenizer,
            512,
        )
        self.trainer = BaseTrainer.create(
            "lightning_trainer",
            self.engine,
            dataset,
            self.collate_fn,
            3,
            8,
            4e-3,
            True,
            True,
        )
        self.trainer.fit()
