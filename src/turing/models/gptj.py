from typing import Optional

from turing.engines.gptj_engine import GPTJEngine
from turing.models.causal import CausalModel


class GPTJ(CausalModel):
    config_name: str = "gptj"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPTJEngine.config_name, weights_path)


class GPTJLORA(GPTJ):
    config_name: str = "gptj_lora"

    def __init__(self, weights_path: Optional[str] = None):
        self.engine = BaseEngine.create("gptj_lora_engine", weights_path)

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
