import json

from datasets import Dataset, DatasetDict

# Convert the alpaca JSON dataset to HF format


# Right now only the HuggingFace datasets are supported, that's why the JSON Alpaca dataset
# needs to be converted to the HuggingFace format. In addition, this HF dataset should have 3 columns for instruction finetuning: instruction, text and target.
def preprocess_alpaca_json_data(alpaca_dataset_path: str):
    """Creates a dataset given the alpaca JSON dataset. You can download it here: https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

    :param alpaca_dataset_path: path of the Alpaca dataset
    """
    alpaca_data = json.load(open(alpaca_dataset_path))
    instructions = []
    inputs = []
    outputs = []

    for data in alpaca_data:
        instructions.append(data["instruction"])
        inputs.append(data["input"])
        outputs.append(data["output"])

    data_dict = {
        "train": {"instruction": instructions, "text": inputs, "target": outputs}
    }

    dataset = DatasetDict()
    # using your `Dict` object
    for k, v in data_dict.items():
        dataset[k] = Dataset.from_dict(v)

    dataset.save_to_disk(str("./alpaca_data"))
