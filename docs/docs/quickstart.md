---
sidebar_position: 2
title: 🚀 Quickstart
description: Your first fine-tuning job with xTuring
---

# Quickstart

**xTuring** provides fast, efficient and simple fine-tuning of LLMs, such as LLaMA, GPT-J, GPT-2, and more. It supports both single GPU and multi-GPU training. Leverage memory-efficient fine-tuning techniques like LoRA to reduce your hardware costs by up to 90% and train your models in a fraction of the time.

Here is a quick example of how to fine-tune LLaMA 7B on the Alpaca dataset using LoRA technique.

### 1. Install

```bash
pip install xturing
```

### 2. Load the dataset

Download the Alpaca dataset from [here](https://d33tr4pxdm6e2j.cloudfront.net/public_content/tutorials/datasets/alpaca_data.zip) and extract it to a folder. We will load this dataset using the `InstructionDataset` class.

```python
from xturing.datasets import InstructionDataset

dataset = InstructionDataset("./alpaca_data")
```

### 3. Initialize the model
You can also load the LLaMA model without LoRA initiliazation or load one of the other models supported by xTuring. Look at the [supported models](/#models-supported) section for more details.

```python
from xturing.models import BaseModel

model = BaseModel.create("llama_lora")
```

### 4. Fine-tune the model

We will use the default configuration for the fine-tuning.

```python
model.finetune(dataset=dataset)
```

### 5. Generate text

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])
```

### 6. Save the model
You can save the model to use it later by calling the `save` method and then passing the path to the folder where you want to save the model.

```python
model.save("llama_lora_finetuned")
```

### 7. Launch the playground

```python
Playground().launch()
```
