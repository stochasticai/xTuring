from pathlib import Path

from xturing.config.read_config import (
    exists_xturing_config_file,
    read_xturing_config_file,
)
from xturing.registry import BaseParent
from xturing.utils.hub import ModelHub


class BaseModel(BaseParent):
    registry = {}

    @classmethod
    def load(cls, weights_dir_or_model_name, model_name=None, **kwargs):
        path_weights_dir_or_model_name = Path(weights_dir_or_model_name)

        if path_weights_dir_or_model_name.is_dir():
            if exists_xturing_config_file(path_weights_dir_or_model_name):
                return cls.load_from_local(path_weights_dir_or_model_name)

            if model_name is None:
                raise ValueError(
                    "No xturing.json found in local directory '{}'. "
                    "Pass model_name=... to BaseModel.load(...) for local non-xTuring "
                    "checkpoints, or use a GenericModel class directly.".format(
                        str(path_weights_dir_or_model_name)
                    )
                )

            return cls._load_local_path_for_model_name(
                path_weights_dir_or_model_name, model_name, **kwargs
            )

        print("Loading model from xTuring hub")
        return cls.load_from_hub(weights_dir_or_model_name)

    @classmethod
    def load_from_hub(cls, model_name):
        hub = ModelHub()
        model_path = hub.load(model_name)
        return cls.load_from_local(model_path)

    @classmethod
    def load_from_local(cls, weights_dir_path):
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

        return cls._load_local_path_for_model_name(weights_dir_path, model_name)

    @classmethod
    def _load_local_path_for_model_name(cls, weights_dir_path, model_name, **kwargs):
        weights_dir_path = Path(weights_dir_path)

        assert (
            cls.registry.get(model_name) is not None
        ), "The model_name {} is not valid".format(model_name)

        if "generic" in model_name:
            model = cls.create(
                model_name,
                model_name=model_name,
                weights_path=weights_dir_path,
                **kwargs,
            )
        else:
            model = cls.create(model_name, weights_path=weights_dir_path, **kwargs)

        return model


# add_to_registry is a class method, so it's called on the class, not on an instance of the class
# registration happens in __init__.py, to avoid circular imports
