---
title: 💾 Load and save models
description: Load and save models
sidebar_position: 6
---

# Load and save models

### 1. Load a pre-trained model

To load a pre-trained model for the first time, run the following line of code. This will load the model with the default weights.

```python
from xturing.models.base import BaseModel

model = BaseModel.create("distilgpt2_lora")
```

### 2. Save a fine-tuned model

After fine-tuning your model, you can save it as simple as:

```python
model.save("/path/to/a/directory")
```

Remember that the path that you specify should be a directory. If the directory doesn't exist, it will be created.

The model weights will be saved into 2 files. The whole model weights including based model parameters and LoRA parameters are stored in `pytorch_model.bin` file and only LoRA parameters are stored in `adapter_model.bin` file.

### 3. Load a fine-tuned model

To load a saved model, you only have to run the `load` method specifying the directory where the weights were saved.

```python
finetuned_model = BaseModel.load("/path/to/a/directory")
```
