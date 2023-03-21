from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.preprocessors.base import BasePreprocessor
from turing.trainers.base import BaseTrainer


class Llama:
    config_name: str = "llama"

    def __init__(self, weights_path: str):
        self.engine = BaseEngine.create("llama_engine", weights_path)

        self.collate_fn = None
        self.trainer = None

    def finetune(self, dataset: Union[TextDataset, InstructionDataset]):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        self.collate_fn = BasePreprocessor.create(
            dataset.config_name, self.engine.tokenizer, 512
        )
        self.trainer = BaseTrainer.create(
            "lightning_trainer", self.engine, dataset, self.collate_fn
        )
        self.trainer.fit()

    def evaluate(self, dataset: Union[TextDataset, InstructionDataset]):
        pass

    def generate(
        self,
        texts: Optional[Union[List[str], str]] = None,
        dataset: Optional[Union[TextDataset, InstructionDataset]] = None,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.engine.model.eval()

        if texts is not None:
            texts = [texts] if isinstance(texts, str) else texts

            outputs = []
            for text in tqdm(texts):
                inputs = self.engine.tokenizer(text, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        output = self.engine.model.generate(
                            input_ids=input_ids, do_sample=False, max_new_tokens=300
                        )

                output = self.engine.tokenizer.decode(
                    output[0], skip_special_tokens=False
                )
                outputs.append(output)

        elif dataset is not None:
            collate_fn = (
                BasePreprocessor.create(dataset.con, self.engine.tokenizer, 512)
                if isinstance(dataset) == TextDataset
                else BasePreprocessor("instruction_dataset")(self.engine.tokenizer, 512)
            )
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

            outputs = []
            for i, batch in enumerate(tqdm(dataloader)):
                input_ids = batch["input_ids"].to(self.device)
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        output = self.engine.model.generate(
                            input_ids=input_ids, do_sample=False, max_new_tokens=300
                        )

                output = self.engine.tokenizer.decode(
                    output[0], skip_special_tokens=False
                )
                outputs.append(output)
        else:
            raise ("Make sure texts or dataset is not None")

        return outputs

    def save(self, path: Union[str, Path]):
        pass


class LlamaLORA(Llama):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create("llama_lora_engine", weights_path)

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
