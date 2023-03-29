from pathlib import Path

from xturing.config.read_config import (
    exists_xturing_config_file,
    read_xturing_config_file,
)
from xturing.registry import BaseParent


class BaseModel(BaseParent):
    registry = {}

    @classmethod
    def load(cls, weights_dir_path):
        weights_dir_path = Path(weights_dir_path)
        assert weights_dir_path.is_dir(), "The path {} should be a directory".format(
            str(weights_dir_path.resolve())
        )

        assert exists_xturing_config_file(
            weights_dir_path
        ), "We were not able to find the xturing.json file in this directory {}. Please, provide a valid directory".format(
            str(weights_dir_path)
        )

        xturing_config = read_xturing_config_file(weights_dir_path)
        model_name = xturing_config.get("model_name")

        assert (
            model_name is not None
        ), "The xturing.json file is not correct. model_name is not available in the configuration"

        assert (
            cls.registry.get(model_name) is not None
        ), "The model_name {} is not valid".format(model_name)

        model = cls.create(model_name, weights_path=weights_dir_path)

        return model


# add_to_registry is a class method, so it's called on the class, not on an instance of the class
# registration happens in __init__.py, to avoid circular imports
