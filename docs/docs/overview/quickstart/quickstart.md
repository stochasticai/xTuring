---
sidebar_position: 2
title: 🚀 Quick Tour
description: Your first fine-tuning job with xTuring
---

<!-- ## Quick Start -->

**xTuring** provides fast, efficient and simple fine-tuning of LLMs, such as LLaMA, GPT-J, GPT-2, and more. It supports both single GPU and multi-GPU training. Leverage memory-efficient fine-tuning techniques like LoRA to reduce your hardware costs by up to 90% and train your models in a fraction of the time.

Whether you are someone who develops AI pipelines for a living or someone who just wants to leverage the power of AI, this quickstart will help you get started with `xTuring` and how to use `BaseModel` for inference, fine-tuning and saving the obtained weights.


Before diving into it, make sure you have the library installed on your machine:
```bash
pip install xturing
```


<!-- ## class `BaseModel` -->
## Load Supported Model

The `BaseModel` is the easiest way use an off-the-shelf supported model for inference and fine-tuning.
You can use `BaseModel` to load from a wide-range of supported models, the list of which is mentioned [here](/supported_models).

<!-- ### Supported Models
|  Model |  Key | Description |
| -- | -- | ---- |
|Bloom | bloom | Bloom 1.1B model |
|Cerebras | cerebras | Cerebras-GPT 1.3B model |
|DistilGPT-2 | distilgpt2 | DistilGPT-2 model |
|Falcon | falcon | Falcon 7B model |
|Galactica | galactica | Galactica 6.7B model |
|GPT-J | gptj | GPT-J 6B model |
|GPT-2 | gpt2 | GPT-2 model |
|LLaMA | llama | LLaMA 7B model |
|LlaMA2 | llama2 | LLaMA2 model |
|OPT | opt | OPT 1.3B model |

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
model = GenericLoraKbitModel('<model_path>')
```
The `model_path` can be replaced with you local directory or any HuggingFace library model like `facebook/opt-1.3b`. -->

In this guide, we will be using `BaseModel` to fine-tune __LLaMA 7B__ on the __Alpaca dataset__ using __LoRA__ technique.

Start by downloading the Alpaca dataset from [here](https://d33tr4pxdm6e2j.cloudfront.net/public_content/tutorials/datasets/alpaca_data.zip) and extract it to a folder. We will load this dataset using the `InstructionDataset` class.

```python
from xturing.datasets import InstructionDataset

dataset_path = './alpaca_data'

dataset = InstructionDataset(dataset_path)
```

Next, initialize the model.
We can also load the LLaMA model without _LoRA_ initiliazation or load one of the other models supported by xTuring. Look at the [supported models](/#basemodel) section for more details.

```python
from xturing.models import BaseModel

model_name = 'llama_lora'

model = BaseModel.create(model_name)
```

To fine-tune the model on the loaded dataset, we will use the default configuration for the fine-tuning.

```python
model.finetune(dataset=dataset)
```

Let's test our fine-tuned model, and make some inference.

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])
```
Print the `output` variable to see the results.

Next, we need to save our fine-tuned model using the `.save()` method. We will send the path of the directory as parameter to the method to save the fine-tuned model.

```python
finetuned_model_path = 'llama_lora_finetuned'

model.save(finetuned_model_path)
```

We can also see our model(s) in action with a beautiful UI by launchung the playground locally.

```python
from xturing.ui.playground import Playground

Playground().launch()
```

<!-- ## class `GenericModel` -->
## Load Any Model
The `GenericModel` class makes it possible to test and fine-tune the models which are not directly available via the `BaseModel` class. Apart from the base class, we can use classes mentioned below to load the models for memory-efficient computations:

| Class Name | Description |
| ---------- | ----------- |
| `GenericModel` |    Loads the normal version of the model     |
| `GenericInt8Model` |    Loads the model ready to fine-tune in __INT8__ precision     |
| `GenericLoraModel` |    Loads the model ready to fine-tune using __LoRA__ technique     |
| `GenericLoraInt8Model` |   Loads the model ready to fine-tune using __LoRA__ technique in __INT8__ precsion        |
| `GenericLoraKbitModel` |   Loads the model ready to fine-tune using __LoRA__ technique in __INT4__ precision         |

Let us circle back to the above example and see how we can replicate the results of the `BaseModel` class.

Start by downloading the Alpaca dataset from [here](https://d33tr4pxdm6e2j.cloudfront.net/public_content/tutorials/datasets/alpaca_data.zip) and extract it to a folder. We will load this dataset using the `InstructionDataset` class.

```python
from xturing.datasets import InstructionDataset

dataset_path = './alpaca_data'

dataset = InstructionDataset(dataset_path)
```


To initialize the model, simply run the following 2 commands:
```python
from xturing.models import GenericModel

model_path = 'aleksickx/llama-7b-hf'

model = GenericLoraModel(model_name)
```
The _'model_path'_ can be a locally saved model and/or any model available on the HuggingFace's [Model Hub](https://huggingface.co/models).

To fine-tune the model on the loaded dataset, we will use the default configuration for the fine-tuning.

```python
model.finetune(dataset=dataset)
```

Let's test our fine-tuned model, and make some inference.

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])
```
Print the `output` variable to see the results.

Next, we need to save our fine-tuned model using the `.save()` method. We will send the path of the directory as parameter to the method to save the fine-tuned model.

```python
finetuned_model_path = 'llama_lora_finetuned'

model.save(finetuned_model_path)
```

We can also see our model(s) in action with a beautiful UI by launchung the playground locally.

```python
from xturing.ui.playground import Playground

Playground().launch()
```
