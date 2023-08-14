---
title: Guide
description: Inferencing guide
sidebar_position: 1
---

# Inference Guide

## Inference via `BaseModel`

Once you have fine-tuned your model, you can run the inferences as simple as follows.

### Using a local model

Start with loading your model from a checkpoint after fine-tuning it.

```python
# Make the ncessary imports
from xturing.models.base import BaseModel
# Load the desired model
model = BaseModel.load("/path/to/local/model")
```

Next, we can run do the inference on our model using the `.generate()` method.

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```
### Using a pretrained model

Start with loading your model with the default weights

```python
# Make the ncessary imports
from xturing.models.base import BaseModel
# Load the desired model
model = BaseModel.create("llama_lora")
```

Next, we can run do the inference on our model using the `.generate()` method.

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```

## Inference via `GenericModel`

Once you have fine-tuned your model, you can run the inferences as simple as follows.

### Using a local model

Start with loading your model from a checkpoint after fine-tuning it.

```python
# Make the ncessary imports
from xturing.modelsimport GenericModel
# Load the desired model
model = GenericModel("/path/to/local/model")
```

Next, we can run do the inference on our model using the `.generate()` method.

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```
### Using a pretrained model

Start with loading your model with the default weights.

```python
# Make the ncessary imports
from xturing.models import GenericModel
# Load the desired model
model = GenericModel("llama_lora")
```

Next, we can run do the inference on our model using the `.generate()` method.

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```
