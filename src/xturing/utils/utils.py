import contextlib
import io
import sys
from pathlib import Path

import yaml


def read_yamls(config_path):
    conf = {}

    with open(config_path) as f:
        conf.update(yaml.safe_load(f))

    return conf


@contextlib.contextmanager
def no_std_out():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def create_temp_directory(path):
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(
            f"{path} directory already exists. Please specify other directory if you don't want to use previous cache."
        )

    return dir_path
