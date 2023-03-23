from xturing.registry import BaseParent


class BaseEngine(BaseParent):
    registry = {}


# add_to_registry is a class method, so it's called on the class, not on an instance of the class
# registration happens in __init__.py, so it's called when the module is imported
