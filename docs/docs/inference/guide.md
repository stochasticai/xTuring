---
title: Guide
description: Inferencing guide
sidebar_position: 1
---

# Inferencing guide

Once you have fine-tuned your model, you can run the inferences as simple as follows.

### 1. (Optional) Load your model

Load your model from a checkpoint after fine-tuning it.

```python
from xturing.models.base import BaseModel

model = BaseModel.load("/dir/path")
```

Load your model with the default weights

```python
from xturing.models.base import BaseModel

model = BaseModel.create("llama_lora")
```

### 2. Run the generate method

```python
output = model.generate(texts=["Why are the LLMs so important?"])
print("Generated output: {}".format(output))
```
