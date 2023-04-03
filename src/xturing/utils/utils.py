import contextlib
import io
import os
import random
import string
import sys
import tempfile
from pathlib import Path

import textract
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


def extract_text_from_directory(directory_path):
    directory_path = Path(directory_path)
    # Check if the input is a directory
    assert (
        directory_path.is_dir(),
        "The path {} should be a directory".format(directory_path.resolve()),
    )

    print(f"Processing directory {directory_path.resolve()}...")

    # Create a temporary directory to store the txt files
    temp_dir = tempfile.mkdtemp()

    # Walk through the directory tree and process each file
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            print(f"Processing file {file}...")
            # Get the file extension and check if it's supported by textract
            file_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)

            # skip files that are supported
            if ext not in {
                ".csv",
                ".doc",
                ".docx",
                ".eml",
                ".epub",
                ".gif",
                ".jpg",
                ".jpeg",
                ".json",
                ".html",
                ".htm",
                ".mp3",
                ".msg",
                ".odt",
                ".ogg",
                ".pdf",
                ".png",
                ".pptx",
                ".rtf",
                ".tiff",
                ".tif",
                ".txt",
                ".wav",
                ".xlsx",
                ".xls",
            }:
                print(f"{ext} extension is not supported")
                continue

            # Extract the text from the file and save it to a txt file
            try:
                text = textract.process(file_path).decode("utf-8")
                txt_filename = name + ".txt"
                txt_filepath = os.path.join(temp_dir, txt_filename)

                # If the txt file already exists, append a random string to its name
                if os.path.exists(txt_filepath):
                    random_suffix = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=8)
                    )
                    txt_filename = name + "_" + random_suffix + ".txt"
                    txt_filepath = os.path.join(temp_dir, txt_filename)

                with open(txt_filepath, "w") as txt_file:
                    txt_file.write(f"{name}\n\n{text}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    print(f"Finished processing directory, the text files are stored in {temp_dir}.")

    return temp_dir
