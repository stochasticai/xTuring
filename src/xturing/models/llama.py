from typing import Optional

from xturing.engines.llama_engine import (
    LLamaEngine,
    LLamaInt8Engine,
    LlamaLoraEngine,
    LlamaLoraInt8Engine,
    LlamaLoraKbitEngine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraKbitModel,
    CausalLoraModel,
    CausalModel,
)


class Llama(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLamaEngine.config_name, weights_path)


class LlamaLora(CausalLoraModel):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraEngine.config_name, weights_path)


class LlamaInt8(CausalInt8Model):
    config_name: str = "llama_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LLamaInt8Engine.config_name, weights_path)


class LlamaLoraInt8(CausalLoraInt8Model):
    config_name: str = "llama_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraInt8Engine.config_name, weights_path)


class LlamaLoraKbit(CausalLoraKbitModel):
    config_name: str = "llama_lora_kbit"

    # def _make_trainer(
    #     self,
    #     dataset: Union[TextDataset, InstructionDataset],
    #     logger: Union[Logger, Iterable[Logger], bool] = True,
    # ):
    #     return BaseTrainer.create(
    #         LightningTrainer.config_name,
    #         self.engine,
    #         dataset,
    #         self._make_collate_fn(dataset),
    #         int(self.finetuning_args.num_train_epochs),
    #         int(self.finetuning_args.batch_size),
    #         float(self.finetuning_args.learning_rate),
    #         self.finetuning_args.optimizer_name,
    #         True,
    #         True,
    #         lora_type=32,
    #         logger=logger,
    #     )

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraKbitEngine.config_name, weights_path)
