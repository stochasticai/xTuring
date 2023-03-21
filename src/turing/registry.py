from typing import Any, Dict


class BaseParent:
    @classmethod
    def add_to_registry(cls, name: str, obj: Any):
        cls.registry[name] = obj

    @classmethod
    def create(cls, class_key, *args, **kwargs):
        return cls.registry[class_key](*args, **kwargs)

    @classmethod
    def __getitem__(cls, key):
        return cls.registry[key]
