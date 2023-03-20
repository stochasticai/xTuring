from typing import Dict, Any

class BaseParent:
    def __init__(self, registry: Dict[str, Any]):
        self.registry = registry

    def add_to_registry(self, name: str, obj: Any):
        self.registry[name] = obj

    def create(self, type, **kwargs):
        return self.registry[type](**kwargs)

    def __getitem__(self, key):
        return self.registry[key]
