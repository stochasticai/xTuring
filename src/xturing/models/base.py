from xturing.registry import BaseParent


class BaseModel(BaseParent):
    registry = {}


# add_to_registry is a class method, so it's called on the class, not on an instance of the class
# registration happens in __init__.py, to avoid circular imports
