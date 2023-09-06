---
sidebar_position: 2
title: 🚀 Quickstart
description: Your first fine-tuning job with xTuring
---

<!-- ## Quick Start -->

**xTuring** provides fast, efficient and simple fine-tuning of LLMs, such as LLaMA, GPT-J, GPT-2, and more. It supports both single GPU and multi-GPU training. Leverage memory-efficient fine-tuning techniques like LoRA to reduce your hardware costs by up to 90% and train your models in a fraction of the time.

Whether you are someone who develops AI pipelines for a living or someone who just wants to leverage the power of AI, this quickstart will help you get started with `xTuring` and how to use `BaseModel` for inference, fine-tuning and saving the obtained weights.


xTuring provides a solution to easily load and fine-tune some pre-trained models in less than 10 lines of code. Fine-tuning a pre-trained large language models (LLMs) comes with a lot of complexity and requires Machine Learning along with domain knowledge. Hence, it quite a non-intuitive task for beginners and business owners who lack practical knowledge in the field. This particular issue has been addressed by xTuring by providing a simple interface understandable with little to no knowledge of Python. It is as short and crisp as:

```python
from xturing.datasets import InstructionDataset
from xturing.models import BaseModel


# Load a model of your choice
model = BaseModel.create('llama_lora')

# Prepare the training data
dataset = InstructionDataset('...')

# Fine-tune the model
model.finetune(dataset=dataset)

# Test the fine-tuned model
output = model.generate(texts=["Why LLM models are becoming so important?"])

# Save the fine-tuned model
model.save("/path/to/a/directory")
```

Please checkout the following steps to fully understand the above code:

- [Load and Save Models](/overview/quickstart/load_save_models)
- [Load and Prepare Data](/overview/quickstart/prepare)
- [Fine-Tune Models](/overview/quickstart/finetune_guide)
- [Inference](/overview/quickstart/inference)
