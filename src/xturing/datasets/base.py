from torch.utils.data import Dataset

from xturing.registry import BaseParent


class BaseDataset(BaseParent, Dataset):
    registry = {}

    @property
    def meta(self):
        return self._meta
