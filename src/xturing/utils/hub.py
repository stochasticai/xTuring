import shutil
import sys
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

            def bar_progress(current, total, width=80):
                progress_message = "Downloading model : %d%% [%d / %d] bytes" % (
                    current / total * 100,
                    current,
                    total,
                )
                sys.stdout.write("\r" + progress_message)
                sys.stdout.flush()

            try:
                wget.download(url, str(zip_filename), bar=bar_progress)

                with ZipFile(zip_filename, "r") as zip_ref:
                    zip_ref.extractall(path=model_dir)

                print(f"Downloaded model {model_name} from {url}")

                # if zip has folder with model, move files to root

                entries = list(model_dir.glob("*"))

                if len(entries) == 1 and entries[0].is_dir():
                    single_folder = entries[0]

                    for item in single_folder.iterdir():
                        shutil.move(str(item), str(model_dir / item.name))

                    shutil.rmtree(single_folder)

            except Exception as e:
                print(f"Error downloading model {model_name} from {url}: {e}")
                raise e
            finally:
                zip_filename.unlink()

        return model_dir


def make_model_url(model_name: str):
    url = "https://d33tr4pxdm6e2j.cloudfront.net/public_content"
    return f"{url}/models/{model_name}.zip"


class ModelHub(Hub):
    static_path_map = {
        "gpt2": make_model_url("gpt2"),
        "gpt2_lora": make_model_url("gpt2_lora"),
        "distilgpt2": make_model_url("distilgpt2"),
        "distilgpt2_lora": make_model_url("distilgpt2_lora"),
        "llama_lora": make_model_url("llama_lora"),
        "distilgpt2_lora_finetuned_alpaca": make_model_url(
            "distilgpt2_lora_finetuned_alpaca"
        ),
        "llama_lora_finetuned_alpaca": make_model_url("llama_lora_finetuned_alpaca"),
    }

    def __init__(self):
        super().__init__("x/", Path("models"))
