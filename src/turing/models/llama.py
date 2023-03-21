from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from turing.datasets.base import BaseDataset
from turing.datasets.instruction_dataset import InstructionDataset
from turing.datasets.text_dataset import TextDataset
from turing.engines.base import BaseEngine
from turing.engines.llama_engine import LLamaEngine, LlamaLoraEngine
from turing.models.causal import CausalLoraModel, CausalModel


class Llama(CausalModel):
    config_name: str = "llama"

    def __init__(self, weights_path: str):
        super().__init__(LLamaEngine.config_name, weights_path)


class LlamaLORA(CausalLoraModel):
    config_name: str = "llama_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(LlamaLoraEngine.config_name, weights_path)
