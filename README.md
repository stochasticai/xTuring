<p align="center">
  <img src=".github/stochastic_logo_light.svg#gh-light-mode-only" width="250" alt="Stochastic.ai"/>
  <img src=".github/stochastic_logo_dark.svg#gh-dark-mode-only" width="250" alt="Stochastic.ai"/>
</p>
<h3 align="center">Efficient, fast, and simple fine-tuning of LLM models</h3>

___

`xturing` is a python package to perform efficient fine-tuning of LLM models like LLaMA, GPT-J, GPT-2 and more. It supports both single GPU and multi-GPU training. Leverage efficient fine-tuning techniques like LoRA to reduce your hardware costs by up to 90% and train your models in a fraction of the time.

<br>

## ⚙️ Installation
```bash
pip install xturing
```

<br>

## 🚀 Quickstart
```python
from xturing.datasets import InstructionDataset
from xturing.models import BaseModel

# Load the dataset
instruction_dataset = InstructionDataset("./alpaca_data")

# Initialize the model
model = BaseModel.create("llama_lora")

# Finetune the model
model.finetune(dataset=instruction_dataset)

# Perform inference
output = model.generate(texts=["Why LLM models are becoming so important?"])

print("Generated output by the model: {}".format(output))
```

You can find the data folder [here](examples/llama_lora_alpaca/alpaca_data/).

<br>


## 📚 Tutorials
- [Preparing your dataset](examples/llama/preparing_your_dataset.py)
- [LLaMA efficient fine-tuning with LoRA](examples/llama/llama_lora.py)
- [LLaMA fine-tuning](examples/llama/llama.py)
- [GPT-J efficient fine-tuning with LoRA](examples/gptj/gptj_lora.py)
- [GPT2 efficient fine-tuning with LoRA](examples/gpt2/gpt2_lora.py)

<br>

## 📈 Roadmap
- [x] Support for LLaMA, GPT-J, GPT-2
- [ ] Support for Stable Diffusion
- [ ] Dataset generation using self-instruction
- [ ] Evaluation of LLM models

<br>

## 🤝 Help and Support
If you have any questions, you can create an issue on this repository.

You can also join our [Discord server](https://discord.gg/TgHXuSJEk6) and start a discussion in the `#xturing` channel.

<br>

## 📝 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

<br>

## 🌎 Contributing
As an open source project in a rapidly evolving field, we welcome contributions of all kinds, including new features and better documentation. Please read our [contributing guide](CONTRIBUTING.md) to learn how you can get involved.
