---
title: 🤖 Adding new model
description: Guide on how to add new models to xTuring
sidebar_position: 3
---

# 🤖 How to add a model to _xTuring_?

We appreciate your interest in contributing new models to xTuring. This guide will help you understand how to add new models and engines to the library.

## Prerequisites
Before we start, we need to make sure that we are familiar with the `xTuring` codebase, particularly the structure of the [__models__](https://github.com/stochasticai/xTuring/tree/main/src/xturing/models) and [__engines__](https://github.com/stochasticai/xTuring/tree/main/src/xturing/engines) folders. Familiarity with _PyTorch_ and the _Hugging Face Transformers_ library is also essential.

## Steps to add a new model

### 1. First, create a new engine
In the `engines` folder, we are required to create a new file with the name of our engine (e.g., `my_engine.py`). Define a new engine class that inherits from an appropriate base engine class (e.g., `CausalEngine` for causal models, for `Seq2Seq` you need to add base class first). Implement necessary methods and attributes, following the structure of existing engine. We can refer [gptj_engine.py](https://github.com/stochasticai/xTuring/blob/main/src/xturing/engines/gptj_engine.py).

```python
from xturing.engines.causal import CausalEngine

class MyEngine(CausalEngine):
    config_name: str = "my_engine"

    def __init__(self, model_name: str, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name, weights_path)
```

### 2. Next, create a new model
In the `models` folder, we need to create a new file with the name of our model (e.g., `my_model.py`). First, define a new model class that inherits from an appropriate base model class (e.g., `CausalModel` for causal models, for `Seq2Seq` you need to add base class first). Then, implement necessary methods and attributes, following the structure of existing models. We can refer [gptj.py](https://github.com/stochasticai/xTuring/blob/main/src/xturing/models/gptj.py).

```python
from xturing.models.causal import CausalModel
from xturing.engines.my_engine import MyEngine

class MyModel(CausalModel):
    config_name: str = "my_model"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MyEngine.config_name, weights_path)
```

### 3. Then, update the model and engine registries 
We have to add our new model to the [`model`](https://github.com/stochasticai/xTuring/blob/main/src/xturing/models/__init__.py) and [`engine`](https://github.com/stochasticai/xTuring/blob/main/src/xturing/engines/__init__.py) registries in `xturing.{models|engines}.__init__.py`. This will allow users to create instances of your model using the `BaseModel.create('<model_name>')` method.

### 4. Alongside, update the config files
In the `config` folder, we need to add our new model key and their respective hyperparameters to be run by default in `finetuning_config.yaml` and `generation_config.yaml` files.
```yaml
# finetuning_config.yaml
my_model:
    learning_rate: 1e-4
    weight_decay: 0.01
    num_train_epochs: 3
    batch_size: 8
    max_length: 256

# generation_config.yaml
my_model:
    max_new_tokens: 256
    do_sample: false

```

### 5. Do not forget to add tests 
If our model is small enough we can add tests for our new model in the [tests/](https://github.com/stochasticai/xTuring/tree/main/tests/xturing) folder. We can use existing tests as a reference. If our model is too large to be included in the tests, we can add a notebook in the [examples/](https://github.com/stochasticai/xTuring/tree/main/examples) folder to demonstrate how to use our model.

### 6. Update the documentation
We should not forget to update the documentation to include our new model and engine. We can do so by adding a new _Markdown_ file in the [examples/](https://github.com/stochasticai/xTuring/tree/main/examples) folder with a tutorial on how to use our model.

### 7. At last, submit a pull request
Once we have completed the above steps, we are ready to submit a pull request to the `dev` branch. For that, we should provide a clear description of our changes and why they are needed. Then the seasoned contributors will review our changes as soon as possible and provide feedback. Once our changes have been approved, they will be merged into the `dev` branch.

## Steps to add a LoRA model

### 1. First, Create a new engine
Analogous to the steps above, create a new engine in the `engines` folder. The new engine should inherit from the `CausalLoraEngine` base class. We can use [`gptj_engine.py`](https://github.com/stochasticai/xTuring/blob/main/src/xturing/engines/gpt2_engine.py) file as a reference.

```python
from xturing.engines.causal import CausalLoraEngine

class MyLoraEngine(CausalLoraEngine):
    config_name: str = "my_engine_lora"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name,
            weights_path,
            target_modules=[],
        )
```

The `target_modules` parameter is the list of identifiers used to denote attention layers of your model. For example, for *GPTJ* model, the attention layers are denoted with `q_proj` and `v_proj`.

### 2. Next, create a new model
Analogous to the steps above, create a new model in the `models` folder. The new model should inherit from the `CausalLoraModel` base class. We can use the [`gptj.py`](https://github.com/stochasticai/xTuring/blob/main/src/xturing/models/gptj.py) file as a reference.
```python
from xturing.models.causal import CausalLoraModel
from xturing.engines.my_engine import MyLoraEngine

class MyModelLora(CausalLoraModel):
    config_name: str = "my_model_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(MyLoraEngine.config_name, weights_path)
```

### Next, follow steps 3 through 7 as mentioned in the above steps

Thank you for your contribution to xTuring!

> Note: For ease of review, make sure to run your model locally with a sample prompt and paste your output(s) along with the sample prompt in the description space of your pull request.
