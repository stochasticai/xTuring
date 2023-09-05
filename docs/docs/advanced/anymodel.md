---
title: 🌦️ Work with any model
description: Use self-instruction to generate a dataset
sidebar_position: 2
---

<!-- ## class `GenericModel` -->
<!-- ## Load Any Model via `GenericModel` wrapper -->
The `GenericModel` class makes it possible to test and fine-tune the models which are not directly available via the `BaseModel` class. Apart from the base class, we can use classes mentioned below to load the models for memory-efficient computations:

| Class Name | Description |
| ---------- | ----------- |
| `GenericModel` |    Loads the normal version of the model     |
| `GenericInt8Model` |    Loads the model ready to fine-tune in __INT8__ precision     |
| `GenericLoraModel` |    Loads the model ready to fine-tune using __LoRA__ technique     |
| `GenericLoraInt8Model` |   Loads the model ready to fine-tune using __LoRA__ technique in __INT8__ precsion        |
| `GenericLoraKbitModel` |   Loads the model ready to fine-tune using __LoRA__ technique in __INT4__ precision         |

<!-- Let us circle back to the above example and see how we can replicate the results of the `BaseModel` class as shown [here](/overview/quickstart/load_save_models). -->

<!-- Start by downloading the Alpaca dataset from [here](https://d33tr4pxdm6e2j.cloudfront.net/public_content/tutorials/datasets/alpaca_data.zip) and extract it to a folder. We will load this dataset using the `InstructionDataset` class. -->

<!-- ```python
from xturing.datasets import InstructionDataset

dataset_path = './alpaca_data'

dataset = InstructionDataset(dataset_path)
``` -->


To initialize the model, simply run the following 2 commands:
```python
from xturing.models import GenericModel

model_path = 'aleksickx/llama-7b-hf'

model = GenericLoraModel(model_path)
```
The _'model_path'_ can be a locally saved model and/or any model available on the HuggingFace's [Model Hub](https://huggingface.co/models).

To fine-tune the model on a dataset, we will use the default configuration for the fine-tuning.

```python
model.finetune(dataset=dataset)
```

In order to see how to load a pre-defined dataset, go [here](/overview/quickstart/prepare), and to see how to generate a dataset, refer [this](/advanced/generate) page.

Let's test our fine-tuned model, and make some inference.

```python
output = model.generate(texts=["Why LLM models are becoming so important?"])
```
We can print the `output` variable to see the results.

Next, we need to save our fine-tuned model using the `.save()` method. We will send the path of the directory as parameter to the method to save the fine-tuned model.

```python
model.save('/path/to/a/directory/')
```

We can also see our model(s) in action with a beautiful UI by launchung the playground locally.

```python
from xturing.ui.playground import Playground

Playground().launch()
```

<!-- ## GenericModel classes
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

## Inference via `GenericModel`

Once you have fine-tuned your model, you can run the inferences as simple as follows.

### Using a local model

Start with loading your model from a checkpoint after fine-tuning it.

```python
# Make the ncessary imports
from xturing.modelsimport GenericModel
# Load the desired model
model = GenericModel("/path/to/local/model")
```

Next, we can run do the inference on our model using the `.generate()` method.

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
```
### Using a pretrained model

Start with loading your model with the default weights.

```python
# Make the ncessary imports
from xturing.models import GenericModel
# Load the desired model
model = GenericModel("llama_lora")
```

Next, we can run do the inference on our model using the `.generate()` method.

```python
# Make inference
output = model.generate(texts=["Why are the LLMs so important?"])
# Print the generated outputs
print("Generated output: {}".format(output))
``` -->
