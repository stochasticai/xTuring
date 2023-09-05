---
title: 🔧 Fine-tune pre-trained models
description: Fine-tuning with xTuring
sidebar_position: 4
---

import Test from './test';

<!-- # Fine-tuning guide -->

## Text fine-tuning
After preparing the dataset in correct format, you can start the text fine-tuning.

First, load the text dataset and initialize the model of your choice

<Test instruction={'Text'} />


Next, we need to start the fine-tuning

```python
model.finetune(dataset=dataset)
```

<!-- Finally, let us test how our fine-tuned model performs using the `.generate()` function.

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])

# Print the model outputs
print("Generated output by the model: {}".format(output))
``` -->


## Instruction fine-tuning

First, make sure that you have prepared your fine-tuning dataset for instruction fine-tuning. To know how, refer [here](/overview/quickstart/prepare#instructiondataset).

After preparing the dataset in correct format, you can start the instruction fine-tuning.

Start by loading the instruction dataset and initializing the model of your choice.

<Test instruction={'Instruction'}/>

A list of all the supported models can be found [here](/overview/supported_models).



Next, we need to start the fine-tuning

```python
model.finetune(dataset=instruction_dataset)
```

<!-- Finally, let us test how our fine-tuned model performs using the `.generate()` function.

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])

# Print the model outputs
print("Generated output by the model: {}".format(output))
``` -->

<!-- xTuring supports following models:

|   Model Name     |      Model Key      | Description |
| ------------ | --------- | ---- |
| BLOOM | bloom | Bloom 1.1B model |
| BLOOM LoRA  | bloom_lora | Bloom 1.1B model with LoRA technique to speed up fine-tuning  |
| BLOOM LoRA INT8 | bloom_lora_int8 | Bloom 1.1B INT8 model with LoRA technique to speed up fine-tuning |
| Cerebras  | cerebras | Cerebras-GPT 1.3B model |
| Cerebras LoRA  | cerebras_lora | Cerebras-GPT 1.3B model with LoRA technique to speed up fine-tuning  |
| Cerebras LoRA INT8  | cerebras_lora_int8 | Cerebras-GPT 1.3B INT8 model with LoRA technique to speed up fine-tuning |
| DistilGPT-2  | distilgpt2 | DistilGPT-2 model |
| DistilGPT-2 LoRA | distilgpt2_lora | DistilGPT-2 model with LoRA technique to speed up fine-tuning  |
| Galactica  | galactica | Galactica 6.7B model |
| Galactica LoRA  | galactica_lora | Galactica 6.7B model with LoRA technique to speed up fine-tuning  |
| Galactica LoRA INT8  | galactica_lora_int8 | Galactica 6.7B INT8 model with LoRA technique to speed up fine-tuning |
| GPT-J | gptj | GPT-J 6B model |
| GPT-J LoRA | gptj_lora | GPT-J 6B model with LoRA technique to speed up fine-tuning  |
| GPT-J LoRA INT8 | gptj_lora_int8 | GPT-J 6B INT8 model with LoRA technique to speed up fine-tuning
| GPT-2 | gpt2 | GPT-2 model |
| GPT-2 LoRA  | gpt2_lora | GPT-2 model with LoRA technique to speed up fine-tuning  |
| GPT-2 LoRA INT8  | gpt2_lora_int8 | GPT-2 INT8 model with LoRA technique to speed up fine-tuning |
| LLaMA | llama | LLaMA 7B model |
| LLaMA LoRa | llama_lora | LLaMA 7B model with LoRA technique to speed up fine-tuning  |
| LLaMA LoRA INT8  | llama_lora_int8 | LLaMA 7B INT8 model with LoRA technique to speed up fine-tuning
| OPT | opt | OPT 1.3B model |
| OPT LoRA | opt_lora | OPT 1.3B model with LoRA technique to speed up fine-tuning  |
| OPT LoRA INT8 | opt_lora_int8 | OPT 1.3B INT8 model with LoRA technique to speed up fine-tuning | -->