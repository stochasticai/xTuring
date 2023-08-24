---
title: 🏋🏻‍♂️ Fine-tuning Configuration
description: Fine-tuning parameters
sidebar_position: 2
---

<!-- # Fine-tuning configuration -->

xTuring is easy to use. The library already loads the best parameters for each model by default.

For advanced usage, you can customize the `finetuning_config` parameter.

### 1. Instantiate your model and dataset

```python
from xturing.models.base import BaseModel
from xturing.datasets.instruction_dataset import InstructionDataset

instruction_dataset = InstructionDataset("alpaca_data")
model = BaseModel.create("llama_lora")
```

### 2. Load the config object

Print the `finetuning_config` object to check the default configuration.

```python
finetuning_config = model.finetuning_config()
```

### 3. Set the config

```python
finetuning_config.batch_size = 64
finetuning_config.num_train_epochs = 1
finetuning_config.learning_rate = 1e-5
finetuning_config.weight_decay = 0.01
finetuning_config.optimizer_name = "adamw"
finetuning_config.output_dir = "training_dir/"
```
#### Parameters

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
| warmup_steps | int | [0,learning_rate] | 50 | The number of steps used for a linear warmup from 0 to learning_rate. |
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

### 4. Start the finetuning

```python
model.finetune(dataset=instruction_dataset)
```