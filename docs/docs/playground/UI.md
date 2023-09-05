---
title: 👁️‍🗨️ UI
description: UI Playground
sidebar_position: 1
---


# 👁️‍🗨️ Exploring the UI playground: a first-person experience

To embark on an interactive journey with the xTuring UI Playground, here's your guide to installation, setup, and seamless engagement with this fascinating tool.

### Prerequisites: Ensuring the latest xTuring version
<!-- Be sure to have the latest version of xturing installed: -->

Begin by guaranteeing that you're equipped with the most up-to-date xTuring version. Execute the subsequent command to ensure the latest update:

```sh
$ pip install xturing --upgrade
```

![Playground UI Demo](/img/playground/ui-playground.gif)

<!-- ### 1. Launch the playground -->
### 1. Launching the playground Interface

To immerse yourself in the world of the UI Playground, you have two equally effective methods:

#### Option 1: Command-Line Interface (CLI)

Execute the command xturing ui in your terminal to launch the UI Playground.

#### Option 2: Integration in a script

Alternatively, in a Python script, you can utilize the following code snippet to launch the Playground interface:

<!-- To launch the playground interface, you can either run `xturing ui` on the CLI or in a script as follows: -->

```python
from xturing.ui.playground import Playground

Playground().launch()
```

:::info
<!-- `Playground` constructor accepts the following argument: -->
For enhanced customization, the Playground constructor accepts the argument model_path, allowing you to specify the model path when launching the Playground:
```
Playground(model_path="...").launch()
```
:::

### 2. Loading the model

Load your desired model effortlessly using one of two methods:

<!-- You can load the model by specifying the model path in the step 1 or by providing the path in the input field. -->

#### Method 1: Path specification (step 1)

During the launch process, provide the model path in Step 1 to initiate model loading.

#### Method 2: Input field (load model section)

Alternatively, you can input the model path directly in the provided field within the UI Playground interface.

![Load model section](/img/playground/load-model.png)

<!-- When you press the load button, model loading will start. Once the model is loaded successfully the chat section is enabled. -->
Upon clicking the "Load" button, the model loading process commences. Once the model is successfully loaded, the chat section becomes active.

:::info
<!-- Model path should be a directory containing a valid `xturing.json` config file. -->
Ensure that the model path points to a directory containing a valid `xturing.json` configuration file.
:::

### 3. Chat with your model

<!-- Enter the prompt and start chatting with the model. Use the `Clear chat` to start a new chat. -->
With the loaded model at your fingertips, enter prompts and initiate engaging conversations with the AI. To start anew, use the "Clear Chat" button for a fresh chat session.

### 4. Tweaking model behavior

<!-- We provide some configuration parameters to change the behavior of the model: `Top-p sampling`, `Contrastive search`. You can change the parameters by using the input fields in the `Parameters` section. -->

The UI Playground offers configuration parameters that enable you to tailor the model's behavior to your preference. In the "Parameters" section, you can adjust settings such as Top-p Sampling and Contrastive Search.

![Parameters section](/img/playground/parameters.png)

Through these intuitive steps, you can readily experience the xTuring UI Playground as if you were navigating it yourself. Uncover the wonders of AI-driven communication and creativity in an interactive, user-friendly environment.