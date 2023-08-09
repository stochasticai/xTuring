---
title: Prepare Dataset
description: Use self-instruction to generate a dataset
sidebar_position: 3
---

## Prepare Instruction dataset

For this tutorial you will need to prepare a dataset which contains 3 columns (instruction, text, target) for instruction fine-tuning or 2 columns (text, target) for text fine-tuning. Here, we show you how to convert Alpaca dataset to be used for instruction fine-tuning.

1. Download Alpaca dataset from this [link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

2. Convert it to instruction dataset format:

```python
import json
from datasets import Dataset, DatasetDict

alpaca_data = json.load(open('/path/to/alpaca_dataset'))
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
for k, v in data_dict.items():
    dataset[k] = Dataset.from_dict(v)

dataset.save_to_disk(str("./alpaca_data"))
```


<!-- :::info

- *alpaca_dataset_path*: The path where the Alpaca dataset is stored.
::: -->

3. Load the prepared Dataset

After preparing the dataset in correct format, you can use this dataset for the instruction fine-tuning.

To load the instruction dataset

```python
from xturing.datasets.instruction_dataset import InstructionDataset

instruction_dataset = InstructionDataset('/path/to/instruction_converted_alpaca_dataset')
```

<!-- :::info

- *dataset_path*: The path where the converted dataset is stored.
::: -->

## Prepare Text dataset

For this, you just need to download a sample dataset to your woring directory, or generate a custom dataset. For example, you can download the Alpaca Dataset from this [link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)
