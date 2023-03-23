from pathlib import Path
from typing import List, Optional, Union

from xturing.datasets.text2image_dataset import Text2ImageDataset


class StableDiffusion:
    config_name: str = "stable_diffusion"

    def __init__(self, weights_path: str):
        pass

    def finetune(self, dataset: Text2ImageDataset):
        pass

    def generate(
        self,
        texts: Optional[Union[List[str], str]] = None,
        dataset: Optional[Text2ImageDataset] = None,
    ):
        pass

    def save(self, path: Union[str, Path]):
        pass
