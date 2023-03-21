from typing import Optional

from turing.engines.gpt2_engine import GPT2Engine

from .causal import CausalModel


class GPT2(CausalModel):
    config_name: str = "gpt2"

    def __init__(
        self, weights_path: Optional[str] = None, engine: str = GPT2Engine.config_name
    ):
        super().__init__(engine, weights_path)


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
