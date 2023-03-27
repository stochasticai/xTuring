from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel


def read_yaml(yml_path: Union[Path, str]):
    yml_path = str(yml_path)
    yml_content = {}

    with open(yml_path) as f:
        yml_content.update(yaml.safe_load(f))

    return yml_content


def load_config(model_name: str, config_path: Union[Path, str], data_class: BaseModel):
    config = {}
    config_path = str(config_path)

    yml_content = read_yaml(config_path)
    # Load default config for all the models
    config.update(yml_content["defaults"])
    # Replace default config params by the specific model params
    config.update(yml_content[model_name])

    config_object = data_class.parse_obj(config)

    return config_object
