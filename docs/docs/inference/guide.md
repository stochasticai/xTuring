---
title: Guide
description: Inferencing guide
sidebar_position: 1
---

# Inference Guide

## Inference using `BaseModel`

Once you have fine-tuned your model, you can run the inferences as simple as follows.

### 1. (Optional) Load your model

Load your model from a checkpoint after fine-tuning it.

```python
# Make the ncessary imports
from xturing.models.base import BaseModel
# Load the desired model
model = BaseModel.load("/dir/path")
```

Load your model with the default weights

```python
# Make the ncessary imports
from xturing.models.base import BaseModel
# Load the desired model
model = BaseModel.create("llama_lora")
```

### 2. Run the generate method

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```

## Inference using `GenericModel`

Once you have fine-tuned your model, you can run the inferences as simple as follows.

### 1. (Optional) Load your model

Load your model from a checkpoint after fine-tuning it.

```python
# Make the necessary imports
from xturing.models import GenericModel
# Load your desired model
model = GenericModel("/dir/path")
```

### 2. Run the generate method

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```
