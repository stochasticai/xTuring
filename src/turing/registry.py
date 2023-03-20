from typing import Any, Dict


class BaseParent:
    def __init__(self, registry: Dict[str, Any]):
        self.registry = registry

    def add_to_registry(self, name: str, obj: Any):
        self.registry[name] = obj

    def create(self, class_key, *args, **kwargs):
        return self.registry[class_key](*args, **kwargs)

    def __getitem__(self, key):
        return self.registry[key]
