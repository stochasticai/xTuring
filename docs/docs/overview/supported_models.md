---
sidebar_position: 4
title: 🦾 Supported models and variants
description: Choose model keys and variant templates for task-based notebooks
---

Use one task notebook from `examples/notebooks/`, then choose a model key from this page.

## Naming templates

| Variant | Key template |
| --- | --- |
| Base | `<model_key>` |
| LoRA | `<model_key>_lora` |
| INT8 | `<model_key>_int8` |
| LoRA + INT8 | `<model_key>_lora_int8` |
| LoRA + K-bit (INT4 flow) | `<model_key>_lora_kbit` |

## Model keys

| Model | Base key | Available variants |
| --- | --- | --- |
| BLOOM 1.1B | `bloom` | `base`, `lora`, `int8`, `lora_int8` |
| Cerebras 1.3B | `cerebras` | `base`, `lora`, `int8`, `lora_int8` |
| DistilGPT-2 | `distilgpt2` | `base`, `lora` |
| Falcon 7B | `falcon` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| Galactica 6.7B | `galactica` | `base`, `lora`, `int8`, `lora_int8` |
| Generic wrapper | `generic` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| GPT-J 6B | `gptj` | `base`, `lora`, `int8`, `lora_int8` |
| GPT-2 | `gpt2` | `base`, `lora`, `int8`, `lora_int8` |
| GPT-OSS 20B | `gpt_oss_20b` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| GPT-OSS 120B | `gpt_oss_120b` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| LLaMA | `llama` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| LLaMA 2 | `llama2` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| Mamba | `mamba` | `base` |
| Mistral 7B | `mistral_7b` | `base` |
| Ministral 3.14B | `ministral_3_14b` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| MiniMaxM2 | `minimax_m2` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| OPT 1.3B | `opt` | `base`, `lora`, `int8`, `lora_int8` |
| Qwen3 0.6B | `qwen3_0_6b` | `base`, `lora`, `int8`, `lora_int8`, `lora_kbit` |
| Stable Diffusion | `stable_diffusion` | `base` |

## INT4-style workflow

For models that expose `*_lora_kbit`, you can still use the generic K-bit API directly:

```python
from xturing.models import GenericLoraKbitModel
model = GenericLoraKbitModel("/path/to/model")
```
