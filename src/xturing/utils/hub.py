from pathlib import Path
from zipfile import ZipFile

import wget


class Hub:
    static_path_map = {}

    def __init__(self, prefix: str, cache_path: Path):
        self.prefix = prefix
        self.cache_path = Path.home() / ".xturing" / cache_path

    def __contains__(self, item):
        return item in self.static_path_map

    def __getitem__(self, item: str):
        if item[: len(self.prefix)] != self.prefix:
            raise ValueError(
                f"Hub path {item} does not start with {self.prefix}, so it is not a valid hub path."
            )
        return item[len(self.prefix) :], self.static_path_map[item[len(self.prefix) :]]

    def load(self, path: str):
        model_name, url = self[path]
        model_dir = self.cache_path / model_name

        if not model_dir.exists():
            print(f"Downloading model {model_name} from {url}")

            self.cache_path.mkdir(parents=True, exist_ok=True)

            zip_filename = self.cache_path / "tmp.zip"

            wget.download(url, str(zip_filename))

            with ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(self.cache_path)

        return model_dir


def make_model_url(model_name: str):
    url = "https://d33tr4pxdm6e2j.cloudfront.net/public_content"
    return f"{url}/models/{model_name}.zip"


class ModelHub(Hub):
    static_path_map = {
        "gpt2": make_model_url("gpt2"),
    }

    def __init__(self):
        super().__init__("x/", Path("models"))
