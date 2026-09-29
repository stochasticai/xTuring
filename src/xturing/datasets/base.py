from datasets import Dataset as HFDataset
from datasets import DatasetDict
from torch.utils.data import Dataset

from xturing.registry import BaseParent


def as_dataset_dict(data):
    """Wrap a bare Hugging Face ``Dataset`` as the ``train`` split of a ``DatasetDict``.

    ``save()`` on the instruction and preference datasets writes only the
    train split, so ``load_from_disk`` hands a plain ``Dataset`` back when a
    saved directory is reloaded. Anything else is returned unchanged.
    """
    if isinstance(data, HFDataset):
        return DatasetDict({"train": data})
    return data


class BaseDataset(BaseParent, Dataset):
    registry = {}

    @property
    def meta(self):
        return self._meta
