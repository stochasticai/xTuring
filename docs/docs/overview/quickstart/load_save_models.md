---
title: 💾 Load and save models
description: Load and save models
sidebar_position: 1
---

# Load and save models

## BaseModel class
### 1. Load a pre-trained model

To load a pre-trained model for the first time, run the following line of code. This will load the model with the default weights.

```python
from xturing.models.base import BaseModel

model = BaseModel.create("<model_key>")

'''
For example,
model = BaseModel.create("distilgpt2_lora")
'''
```
You can find all the supported model keys [here](/supported_models)

### 2. Save a fine-tuned model

After fine-tuning your model, you can save it as simple as:

```python
model.save("/path/to/a/directory")
```

Remember that the path that you specify should be a directory. If the directory doesn't exist, it will be created.

The model weights will be saved into 2 files. The whole model weights including based model parameters and LoRA parameters are stored in `pytorch_model.bin` file and only LoRA parameters are stored in `adapter_model.bin` file.

### 3. Load a fine-tuned model

To load a saved model, you only have to run the `load` method specifying the directory where the weights were saved.

```python
finetuned_model = BaseModel.load("/path/to/a/directory")
```

<details>
<summary><h3>Sample code to load and save a model</h3></summary>

```python
## make the necessary imports
from xturing.models.base import BaseModel

## loading the model
model = BaseModel.create("distilgpt2_lora")

# saving the model
model.save("/path/to/a/directory")

## loading the fine-tuned model
finetuned_model = BaseModel.load("/path/to/a/directory")
```
</details>

## GenericModel classes
The `GenericModel` classes consists of:
1. `GenericModel`
2. `GenericInt8Model`
3. `GenericLoraModel`
4. `GenericLoraInt8Model`
5. `GenericLoraKbitModel`

The below pieces of code will work for all of the above classes by replacing the `GenericModel` in below codes with any of the above classes. The pieces of codes presented below are very similar to that mentioned above with only slight difference.

### 1. Load a pre-trained and/or fine-tuned model

To load a pre-trained (or fine-tuned) model, run the following line of code. This will load the model with the default weights in the case of a pre-trained model, and the weights which were saved in the case of a fine-tuned one.
```python
from xturing.models import GenericModel

model = GenericModel("<model_path>")
'''
The <model_path> can be path to a local model, for example, "./saved_model" or path from the HuggingFace library, for example, "facebook/opt-1.3b"

For example,
model = GenericModel('./saved_model')
OR
model = GenericModel('facebook/opt-1.3b')
'''
```

### 2. Save a fine-tuned model

After fine-tuning your model, you can save it as simple as:

```python
model.save("/path/to/a/directory")
```

Remember that the path that you specify should be a directory. If the directory doesn't exist, it will be created.

The model weights will be saved into 2 files. The whole model weights including based model parameters and LoRA parameters are stored in `pytorch_model.bin` file and only LoRA parameters are stored in `adapter_model.bin` file.


<details>
    <summary> <h3> Examples to load fine-tuned and pre-trained models</h3> </summary>

1. To load a pre-trained model

```python
## Make the necessary imports
from xturing.models import GenericModel

## Loading the model
model = GenericModel("facebook/opt-1.3b")

## Saving the model
model.save("/path/to/a/directory")
```

2. To load a fine-tuned model
```python
## Make the necessary imports
from xturing.models import GenericModel

## Loading the model
model = GenericModel("./saved_model")

```

</details>
