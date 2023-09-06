---
title: 👨🏻‍🏫 Inference
description: Inference parameters
sidebar_position: 2
---

import InferenceCode from './inference_code';

<!-- # Inference configuration   -->

xTuring is easy to use. The library already loads the best parameters for each model by default.

For advanced usage, you can customize the `generation_config` attribute of the model object.

<!-- ## `BaseModel` usage -->

In this tutorial, we will be loading one of the [supported models](/overview/supported_models) and customizing it's generation configuration before running inference.

### Load the model

First, we need to load the model we want to use.

<InferenceCode />

### Load the config object
Next, we need to fetch model's generation configuration using the below command.

```python
generation_config = model.generation_config()
```

We can print the `generation_config` object to check the default configuration.

### Customize the configuration

Now, we can customize the generation configuration as we wish. All the customizable parameters are list [below](#parameters). 

```python
generation_config.max_new_tokens = 256
```

### Test the model
Lastly, we can run inference using the below command to see how our set configuration works.

```python
output = model.generate(texts=["Why are the LLM models important?"])
```
We can print the `output` object to see the results.

### Parameters

<!-- >__max_new_tokens__: the maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
>
> __penalty_alpha__: for contrastive search decoding. The values balance the model confidence and the degeneration penalty.
>
>__top_k__: for contrastive search and sampling decoding method. The number of highest probability vocabulary tokens to keep for top-k-filtering.
>
> __do_sample__: whether or not to use sampling.
>
> __top_p__: for sampling decoding method. If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. -->

| Name | Type | Range | Default | Desription |
| ---  | ---  | ----- | ------- | ---------- |
| max_new_tokens | int | ≥1 | 256 |  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. |
| penalty_alpha | int | [0,1) | 0.6 |  For contrastive search decoding. The values balance the model confidence and the degeneration penalty. |
| top_k | float | ≥0 | 4 |  For contrastive search and sampling decoding method. The number of highest probability vocabulary tokens to keep for  top-k-filtering. |
| do_sample | bool | {true, false} | false |  Whether or not to use sampling. |
| top_p | float | ≥0 | 0 |  For sampling decoding method. If set to float < 1, only the smallest set of most probable tokens with probabilities that add  up to top_p or higher are kept for generation. |

<!-- penalty_alpha: 0.6
top_k: 4
max_new_tokens: 256
do_sample: false -->

<!-- ## `GenericModel` usage

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
> __top_p__: for sampling decoding method. If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. -->
