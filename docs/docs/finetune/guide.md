---
title: Guide
description: Fine-tuning with xTuring
sidebar_position: 1
---

# Fine-tuning guide

## 1. Prepare dataset

For this tutorial you will need to prepare a dataset which contains 3 columns (instruction, text, target) for instruction fine-tuning or 2 columns (text, target) for text fine-tuning. Here, we show you how to convert Alpaca dataset to be used for instruction fine-tuning.

1. Download Alpaca dataset from this [link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

2. Convert it to instruction dataset format:

```python
import json
from datasets import Dataset, DatasetDict

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
for k, v in data_dict.items():
    dataset[k] = Dataset.from_dict(v)

dataset.save_to_disk(str("./alpaca_data"))
```


:::info

- *alpaca_dataset_path*: The path where the Alpaca dataset is stored.
:::

## 2. Instruction fine-tuning

After preparing the dataset in correct format, you can start the instruction fine-tuning.

1. Load the instruction dataset and initialize the model

```python
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.models import BaseModel

instruction_dataset = InstructionDataset(dataset_path)
model = BaseModel.create(model_name)
```

:::info

- *dataset_path*: The path where the converted dataset is stored.
- *model_name*: The model name you want to perform instruction fine-tuning.
:::

xTuring supports following models:

|      Model name      | Description |
| --------- | ---- |
| llama | LLaMA 7B model |
| llama_lora | LLaMA 7B model with LoRA technique to speed up fine-tuning  |
| llama_lora_int8 | LLaMA 7B INT8 model with LoRA technique to speed up fine-tuning
| gptj | GPT-J 6B model |
| gptj_lora | GPT-J 6B model with LoRA technique to speed up fine-tuning  |
| gptj_lora_int8 | GPT-J 6B INT8 model with LoRA technique to speed up fine-tuning
| gpt2 | GPT-2 model |
| gpt2_lora | GPT-2 model with LoRA technique to speed up fine-tuning  |
| gpt2_lora_int8 | GPT-2 INT8 model with LoRA technique to speed up fine-tuning |
| distilgpt2 | DistilGPT-2 model |
| distilgpt2_lora | DistilGPT-2 model with LoRA technique to speed up fine-tuning  |
| opt | OPT 1.3B model |
| opt_lora | OPT 1.3B model with LoRA technique to speed up fine-tuning  |
| opt_lora_int8 | OPT 1.3B INT8 model with LoRA technique to speed up fine-tuning |
| cerebras | Cerebras-GPT 1.3B model |
| cerebras_lora | Cerebras-GPT 1.3B model with LoRA technique to speed up fine-tuning  |
| cerebras_lora_int8 | Cerebras-GPT 1.3B INT8 model with LoRA technique to speed up fine-tuning |
| galactica | Galactica 6.7B model |
| galactica_lora | Galactica 6.7B model with LoRA technique to speed up fine-tuning  |
| galactica_lora_int8 | Galactica 6.7B INT8 model with LoRA technique to speed up fine-tuning |
| bloom | Bloom 1.1B model |
| bloom_lora | Bloom 1.1B model with LoRA technique to speed up fine-tuning  |
| bloom_lora_int8 | Bloom 1.1B INT8 model with LoRA technique to speed up fine-tuning |


2. Start the fine-tuning

```python
model.finetune(dataset=instruction_dataset)
```

3. Generate an output text with the fine-tuned model

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
```

## 3. Text fine-tuning
After preparing the dataset in correct format, you can start the text fine-tuning.

1. Load the text dataset and initialize the model

```python
from xturing.datasets.text_dataset import TextDataset
from xturing.models import BaseModel

instruction_dataset = TextDataset(dataset_path)
model = BaseModel.create(model_name)
```

:::info

- *dataset_path*: The path where the converted dataset is stored.
- *model_name*: The model name you want to perform instruction fine-tuning.
:::

xTuring supports following models:

|      Model name      | Description |
| --------- | ---- |
| llama | LLaMA 7B model |
| llama_lora | LLaMA 7B model with LoRA technique to speed up fine-tuning  |
| llama_lora_int8 | LLaMA 7B INT8 model with LoRA technique to speed up fine-tuning
| gptj | GPT-J 6B model |
| gptj_lora | GPT-J 6B model with LoRA technique to speed up fine-tuning  |
| gptj_lora_int8 | GPT-J 6B INT8 model with LoRA technique to speed up fine-tuning
| gpt2 | GPT-2 model |
| gpt2_lora | GPT-2 model with LoRA technique to speed up fine-tuning  |
| gpt2_lora_int8 | GPT-2 INT8 model with LoRA technique to speed up fine-tuning |
| distilgpt2 | DistilGPT-2 model |
| distilgpt2_lora | DistilGPT-2 model with LoRA technique to speed up fine-tuning  |
| opt | OPT 1.3B model |
| opt_lora | OPT 1.3B model with LoRA technique to speed up fine-tuning  |
| opt_lora_int8 | OPT 1.3B INT8 model with LoRA technique to speed up fine-tuning |
| cerebras | Cerebras-GPT 1.3B model |
| cerebras_lora | Cerebras-GPT 1.3B model with LoRA technique to speed up fine-tuning  |
| cerebras_lora_int8 | Cerebras-GPT 1.3B INT8 model with LoRA technique to speed up fine-tuning |
| galactica | Galactica 6.7B model |
| galactica_lora | Galactica 6.7B model with LoRA technique to speed up fine-tuning  |
| galactica_lora_int8 | Galactica 6.7B INT8 model with LoRA technique to speed up fine-tuning |
| bloom | Bloom 1.1B model |
| bloom_lora | Bloom 1.1B model with LoRA technique to speed up fine-tuning  |
| bloom_lora_int8 | Bloom 1.1B INT8 model with LoRA technique to speed up fine-tuning |


2. Start the fine-tuning

```python
model.finetune(dataset=instruction_dataset)
```

3. Generate an output text with the fine-tuned model

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])
print("Generated output by the model: {}".format(output))
```
