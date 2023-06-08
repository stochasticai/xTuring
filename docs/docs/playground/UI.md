---
title: UI
description: UI Playground
sidebar_position: 1
---


# UI Playground

Be sure to have the latest version of xturing installed:

```sh
pip install xturing --upgrade
```

![Playground UI Demo](/img/playground/ui-playground.gif)

### 1. Launch the playground

To launch the playground interface, you can either run `xturing ui` on the CLI or in a script as follows:

```python
from xturing.ui.playground import Playground

Playground().launch()
```

:::info
`Playground` constructor accepts the following argument:
```
Playground(model_path="...").launch()
```
:::

### 2. Load the model

You can load the model by specifying the model path in the step 1 or by providing the path in the input field.

![Load model section](/img/playground/load-model.png)

When you press the load button, model loading will start. Once the model is loaded successfully the chat section is enabled.

:::info
Model path should be a directory containing a valid `xturing.json` config file.
:::

### 3. Chat with your model

Enter the prompt and start chatting with the model. Use the `Clear chat` to start a new chat.

### 4. Change the parameters

We provide some configuration parameters to change the behavior of the model: `Top-p sampling`, `Contrastive search`. You can change the parameters by using the input fields in the `Parameters` section.

![Parameters section](/img/playground/parameters.png)
