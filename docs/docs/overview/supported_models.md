---
sidebar_position: 4
title: 🦾 Supported models
description: Models Supported by xTuring
---

<!-- # Models supported by xTuring -->
## Base versions
|   Model | Model Key | LoRA | INT8 | LoRA + INT8 | LoRA + INT4 |
| ------ | --- | :---: | :---: | :---: | :---: |
| BLOOM 1.1B| bloom |  ✅ | ✅ | ✅ | ✅ |
| Cerebras 1.3B| cerebras | ✅  | ✅ | ✅ | ✅ |
| DistilGPT-2 | distilgpt2 | ✅  | ✅ | ✅ | ✅ |
| Falcon 7B | falcon | ✅  | ✅ | ✅ | ✅ |
| Galactica 6.7B| galactica | ✅  | ✅ | ✅ | ✅ |
| GPT-J  6B | gptj | ✅ | ✅ | ✅ | ✅ |
| GPT-2  | gpt2 | ✅  | ✅ | ✅ | ✅ |
| LLaMA  7B | llama | ✅ | ✅ | ✅ | ✅ |
| LLaMA2  | llama2 | ✅ | ✅ | ✅ | ✅ |
| MiniMaxM2 | minimax_m2 | ✅ | ✅ | ✅ | ✅ |
| Qwen3 0.6B | qwen3_0_6b | ✅ | ✅ | ✅ | ✅ |
| OPT 1.3B  | opt | ✅ | ✅ |  ✅ | ✅ |

### Memory-efficient versions
> The above mentioned are the base variants of the LLMs. Below are the templates to get their `LoRA`, `INT8`, `INT8 + LoRA` and `INT4 + LoRA` versions.

| Version | Template |
| -- | -- |
| LoRA |  <model_key>_lora|
| INT8 |  <model_key>_int8|
| INT8 + LoRA |  <model_key>_lora_int8|

### INT4 Precision model versions
> In order to load any model's __`INT4+LoRA`__ version, you will need to make use of `GenericLoraKbitModel` class from `xturing.models`. Below is how to use it:
```python
from xturing.models import GenericLoraKbitModel
model = GenericLoraKbitModel('/path/to/model')
```
The `/path/to/model` can be replaced with you local directory or any HuggingFace library model like `facebook/opt-1.3b`.
