---
title: 🏋🏻‍♂️ Fine-tuning
description: Fine-tuning parameters
sidebar_position: 2
---

import FinetuneCode from './fine_tune_code';

<!-- # Fine-tuning configuration -->

xTuring is easy to use. The library already loads the best parameters for each model by default.

For advanced usage, you can customize the `finetuning_config` attribute of the model object.

In this tutorial, we will be loading one of the [supported models](/overview/supported_models) and customizing it's fine-tune configuration before calibrating the model to the desired dataset.

### Load the model and the dataset
First, we need to load the model and the dataset we want to use. 

<FinetuneCode />


### Load the config object

Next, we need to fetch model's fine-tune configuration using the below command.

```python
finetuning_config = model.finetuning_config()
```

Print the `finetuning_config` object to check the default configuration.

### Customize the configuration
Now, we can customize the generation configuration as we wish. All the customizable parameters are list [below](#parameters). 

```python
finetuning_config.batch_size = 64
finetuning_config.num_train_epochs = 1
finetuning_config.learning_rate = 1e-5
finetuning_config.weight_decay = 0.01
finetuning_config.optimizer_name = "adamw"
finetuning_config.output_dir = "training_dir/"
```
### Start the fine-tuning
Now, we can run tune-up the model on our dataset to see how our set configuration works.

```python
model.finetune(dataset=instruction_dataset)
```

### Parameters

<!-- - `learning_rate`: the initial learning rate for the optimizer.
- `gradient_accumulation_steps`: number of updates steps to accumulate the gradients for, before performing a backward/update pass.
- `batch_size`: the batch size per device (GPU/TPU core/CPU…) used for training.
- `weight_decay`: the weight decay to apply to all layers except all bias and LayerNorm weights in the optimizer.
- `warmup_steps`: number of steps used for a linear warmup from 0 to learning_rate.
- `eval_steps`: number of update steps between two evaluations
- `save_steps`: number of updates steps before two checkpoint saves
- `max_length`: the maximum length when tokenizing the inputs.
- `num_train_epochs`: total number of training epochs to perform
- `logging_steps`: number of update steps between two logs
- `max_grad_norm`: maximum gradient norm (for gradient clipping)
- `save_total_limit`: if a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir.
- `optimizer_name`: optimizer that will be used
- `output_dir`: the output directory where the model predictions and checkpoints will be written. -->

| Name | Type | Range | Default | Desription |
| ---  | ---  | ----- | ------- | ---------- |
| learning_rate | float | >0 | 1e-5 | The initial learning rate for the optimizer. |
| gradient_accumulation_steps | int | ≥1 | 1 | The number of updates steps to accumulate the gradients for, before performing a backward/update pass. |
| batch_size | int | ≥1 | 1 | The batch size per device (GPU/TPU core/CPU…) used for training. |
| weight_decay | float | ≥0 | 0.00 | The weight decay to apply to all layers except all bias and LayerNorm weights in the optimizer. |
| warmup_steps | int | ≥0 | 50 | The number of steps used for a linear warmup from 0 to learning_rate. |
| max_length | int | ≥1 | 512 | The maximum length when tokenizing the inputs. |
| num_train_epochs | int | ≥1 | 1 | The total number of training epochs to perform. |
| eval_steps | int | ≥1 | 5000 | The number of update steps between two evaluations. |
| save_steps | int | ≥1 | 5000 | The number of update steps before two checkpoint saves. |
| logging_steps | int | ≥1 | 10 | The number of update steps between two logs. |
| max_grad_norm | float | ≥0 | 2.0 | The maximum gradient norm (for gradient clipping). |
| save_total_limit | int | ≥1 | 4 | If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir. |
| optimizer_name | string | N/A | adamw | The optimizer to be used. |
| output_dir | string | N/A | saved_model | The output directory where the model predictions and checkpoints will be written. |



<!-- learning_rate: 1e-5
gradient_accumulation_steps: 1
batch_size: 1
weight_decay: 0.00
warmup_steps: 50
eval_steps: 5000
save_steps: 5000
max_length: 512
num_train_epochs: 1
logging_steps: 10
max_grad_norm: 2.0
save_total_limit: 4
optimizer_name: adamw
output_dir: saved_model -->

