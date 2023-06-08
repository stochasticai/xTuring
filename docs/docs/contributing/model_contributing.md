---
title: Adding new models
description: Guide on how to add new models to xTuring
sidebar_position: 2
---

# Adding new models

We appreciate your interest in contributing new models to xTuring. This guide will help you understand how to add new models and engines to the library.

## Prerequisites
Before you start, make sure you are familiar with the xTuring codebase, particularly the structure of the models and engines folders. Familiarity with PyTorch and the Hugging Face Transformers library is also essential.

## Steps to add a new model

1. **Create a new engine**: In the `engines` folder, create a new file with the name of your engine (e.g., `my_engine.py`). Define a new engine class that inherits from an appropriate base engine class (e.g., `CausalEngine` for causal models, for `Seq2Seq` you need to add base class first). Implement necessary methods and attributes, following the structure of existing engines (e.g., `gptj_engine.py`).

    ```python
    from xturing.engines.causal import CausalEngine

    class MyEngine(CausalEngine):
        config_name: str = "my_engine"

        def __init__(self, model_name: str, weights_path: Optional[Union[str, Path]] = None):
            super().__init__(model_name, weights_path)
    ```

2. **Create a new model**: In the `models` folder, create a new file with the name of your model (e.g., `my_model.py`). Define a new model class that inherits from an appropriate base model class (e.g., `CausalModel` for causal models, for `Seq2Seq` you need to add base class first). Implement necessary methods and attributes, following the structure of existing models (e.g., `gptj.py`).

    ```python
    from xturing.models.causal import CausalModel
    from xturing.engines.my_engine import MyEngine

    class MyModel(CausalModel):
        config_name: str = "my_model"

        def __init__(self, weights_path: Optional[str] = None):
            super().__init__(MyEngine.config_name, weights_path)
    ```

3. **Update the model registry**: Add your new model to the `model` and `engine` registry in `xturing.{models|engines}.__init__.py`. This will allow users to create instances of your model using the BaseModel.create() method.
4. **Add tests**: If your model is small enough you can add tests for your new model in the `tests` folder. You can use existing tests as a reference. If your model is too large to be included in the tests, you can add a notebook in the `examples` folder to demonstrate how to use your model.
5. **Update the documentation**: Update the documentation to include your new model and engine. Add a new Markdown file in the docs folder with a tutorial on how to use your model.
6. **Submit a pull request**: Once you have completed the above steps, submit a pull request to the dev branch. Provide a clear description of your changes and why they are needed. We will review your changes as soon as possible and provide feedback. Once your changes have been approved, they will be merged into the dev branch.

## Steps to add a LoRA model

1. **Create a new engine**: analogously to the steps above, create a new engine in the `engines` folder. The new engine should inherit from the `CausalLoraEngine` base class. You can use the `gptj_engine.py` file as a reference.
2. **Create a new model**: analogously to the steps above, create a new model in the `models` folder. The new model should inherit from the `CausalLoraModel` base class. You can use the `gptj.py` file as a reference.

Thank you for your contribution to xTuring!
