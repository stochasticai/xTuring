---
title: 💾 Load and save models
description: Load and save models
sidebar_position: 1
---

<!-- # Load and Save models -->

<!-- ## BaseModel class -->
### Load a pre-trained model

To load a pre-trained model for the first time, run the following line of code. This will load the model with the default weights.

```python
from xturing.models.base import BaseModel

model = BaseModel.create("<model_key>")

'''
For example,
model = BaseModel.create("llama_lora")
'''
```
You can find all the supported model keys [here](/overview/supported_models).

### Save a fine-tuned model

After fine-tuning your model, you can save it as simple as:

```python
model.save("/path/to/a/directory")
```

Remember that the path that you specify should be a directory. If the directory doesn't exist, it will be created.

The model weights will be saved into 2 files. The whole model weights including based model parameters and LoRA parameters are stored in `pytorch_model.bin` file and only LoRA parameters are stored in `adapter_model.bin` file.

### Load a model from local directory

To load a saved model, you only have to run the `load` method specifying the directory where the weights were saved.

```python
model = BaseModel.load("/path/to/a/directory")
```

<details>
<summary>Sample code to load and save a model</summary>

```python
from xturing.models.base import BaseModel

## Load the model
model = BaseModel.create("llama_lora")

# Save the model
model.save("/path/to/a/directory")

## Load the fine-tuned model
finetuned_model = BaseModel.load("/path/to/a/directory")
```
</details>

<!-- ## Load Supported Model

The `BaseModel` is the easiest way use an off-the-shelf supported model for inference and fine-tuning.
You can use `BaseModel` to load from a wide-range of supported models, the list of which is mentioned [here](/supported_models).


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

# choose a model 
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
``` -->
