---
title: Inference
description: Inference parameters
sidebar_position: 2
---

# Inference configuration

xTuring is easy to use. The library already loads the best parameters for each model by default.

For advanced usage, you can customize the `.generation_config` attribute of the model.

## `BaseModel` usage

In this tutorial, we will be loading __OPT 1.3B__ model and customizing it's generation configuration before ineferencing. To use any other model, head to [supported models](/overview/supported_models) page for model keys to supported models of `xTuring`.

### 1. Load the model

```python
from xturing.models.base import BaseModel

model = BaseModel.create("opt")
```

### 2. Load the config object

```python
generation_config = model.generation_config()
```
Print the `generation_config` object to check the default configuration.

### 3. Customize the configuration

```python
generation_config.max_new_tokens = 256
```

### 4. Test the model

```python
output = model.generate(texts=["Why are the LLM models important?"])
```
Print the `output` object to see the results.

#### Parameters

>__max_new_tokens__: the maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
>
> __penalty_alpha__: for contrastive search decoding. The values balance the model confidence and the degeneration penalty.
>
>__top_k__: for contrastive search and sampling decoding method. The number of highest probability vocabulary tokens to keep for top-k-filtering.
>
> __do_sample__: whether or not to use sampling.
>
> __top_p__: for sampling decoding method. If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.

## `GenericModel` usage

In this tutorial, we will be loading [__facebook/opt1.3B__](https://huggingface.co/facebook/opt-1.3b) model and customizing it's generation configuration before ineferencing.

### 1. Load the model

```python
from xturing.models.base import BaseModel

model = GenericModel("facebook/opt-1.3B")
```

### 2. Load the config object

```python
generation_config = model.generation_config()
```
Print the `generation_config` object to check the default configuration.

### 3. Customize the configuration

```python
generation_config.max_new_tokens = 256
```

### 4. Test the model

```python
output = model.generate(texts=["Why are the LLM models important?"])
```
Print the `output` object to see the results.

#### Parameters

> __max_new_tokens__: the maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
>
> __penalty_alpha__: for contrastive search decoding. The values balance the model confidence and the degeneration penalty.
>
> __top_k__: for contrastive search and sampling decoding method. The number of highest probability vocabulary tokens to keep for top-k-filtering.
>
> __do_sample__: whether or not to use sampling.
>
> __top_p__: for sampling decoding method. If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
