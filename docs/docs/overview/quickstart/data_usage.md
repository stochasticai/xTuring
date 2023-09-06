---
title: 📜 Use datasets
description: Using a an existing dataset
sidebar_position: 3
---

<!-- ## Prepare Instruction Dataset -->
<!-- There are times, when we want to use some existing dataset for fine-tuning or testing our model on. In such scenarios, it is important to know the format in which we get the dataset and to make sure it is in the format in which `xTuring` will accept for seamless working. If the dataset we choose in a format not expected by `xTuring`, then below is a way to do it. Let's dive right into it! -->

Certainly, when we're looking to utilize an existing dataset for tasks like fine-tuning or model testing, it becomes crucial to ensure that the dataset is in a format compatible with `xTuring`. This ensures smooth functionality when working with the `xTuring` platform. However, there might be instances where the chosen dataset isn't in the format that `xTuring` expects. In such cases, there's a method we can follow to rectify this issue. Let's explore the process:

1. **Selecting the Dataset**: The first step involves choosing a dataset that suits our requirements for fine-tuning or testing the model.

2. **Format Alignment**: It's essential to verify whether the chosen dataset is in the format that `xTuring` accepts. If it's not, we need to proceed with some adjustments.

3. **Format Adjustment**: To align the dataset with `xTuring`'s expectations, we should reformat it according to the accepted structure. This ensures that the platform can seamlessly work with the data.

4. **Ensuring Coherency**: The reformatted text should maintain coherency and clarity. It should effectively convey the message while adhering to proper grammar and organization.

By following these steps, we ensure that the chosen dataset is transformed into a compatible format for `xTuring`, enabling efficient usage and optimal results. 

We know __what__ all we need to do to make format the dataset, below is the __how__ behind it!

## Instruction dataset format
For this tutorial we will need to prepare a dataset which contains 3 columns (instruction, text, target) for instruction fine-tuning or 2 columns (text, target) for text fine-tuning. Here, we will see how to convert Alpaca dataset to be used for instruction fine-tuning. Before starting, make sure you have downloaded the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) in your working directory. 


### Convert the dataset to _Instruction Dataset_ format
This is the main step where we our knowledge of the existing dataset, we convert it to a format understandable by `xTuring`'s _InstructionDataset_ class.

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


### Load the prepared dataset

After preparing the dataset in correct format, you can use this dataset for the instruction fine-tuning.

To load the instruction dataset

```python
from xturing.datasets.instruction_dataset import InstructionDataset

instruction_dataset = InstructionDataset('/path/to/instruction_converted_alpaca_dataset')
```

## Text dataset format

The datasets that we find on the internet are formatted in a way which is accepted by the `xTuring`'s `TextDataset` class, so we need not worry text fine-tuning and just use those datasets as is. 