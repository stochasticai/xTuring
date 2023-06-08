---
title: Configure
description: Inference parameters
sidebar_position: 2
---

# Inference configuration

xTuring is easy to use. The library already loads the best parameters for each model by default.

For advanced usage, you can customize the `generate` method.

### 1. Instantiate your model

```python
from xturing.models.base import BaseModel

model = BaseModel.create("llama_lora")
```

### 2. Load the config object

Print the `generation_config` object to check the default configuration.

```python
generation_config = model.generation_config()
print(generation_config)
```

### 3. Set the config

```python
generation_config.max_new_tokens = 256
```

### 4. Start generating text

```python
output = model.generate(texts=["Why are the LLM models important?"])
print(output)
```

## Reference

- `max_new_tokens`: the maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
- `penalty_alpha`: for contrastive search decoding. The values balance the model confidence and the degeneration penalty.
- `top_k`: for contrastive search and sampling decoding method. The number of highest probability vocabulary tokens to keep for top-k-filtering.
- `do_sample`: whether or not to use sampling.
- `top_p`: for sampling decoding method. If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
