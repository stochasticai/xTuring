from pathlib import Path
from typing import Union


class Text2ImageDataset:
    config_name: str = "text2image_dataset"

    def __init__(self, path: Union[str, Path]):
        raise NotImplementedError("Text2ImageDataset is not implemented yet.")

    def _validate(self):
        raise NotImplementedError
