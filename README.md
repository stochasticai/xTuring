<p align="center">
  <img src=".github/stochastic_logo_light.svg#gh-light-mode-only" width="250" alt="Stochastic.ai"/>
  <img src=".github/stochastic_logo_dark.svg#gh-dark-mode-only" width="250" alt="Stochastic.ai"/>
</p>
<h3 align="center">Build and control your own LLMs</h3>

___

`xturing` provides fast, efficient and simple fine-tuning of LLMs, such as LLaMA, GPT-J, GPT-2,
and more. It supports both single GPU and multi-GPU training. Leverage memory-efficient fine-tuning
techniques like LoRA to reduce your hardware costs by up to 90% and train your models in a fraction
of the time.

<br>

Documentation -  [https://xturing.stochastic.ai/](https://xturing.stochastic.ai/)

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

You can find the data folder [here](examples/llama/alpaca_data).

<br>


## 📚 Tutorials
- [Preparing your dataset](examples/llama/preparing_your_dataset.py)
- [LLaMA efficient fine-tuning with LoRA and INT8](examples/llama/llama_lora_int8.py) &ensp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SQUXq1AMZPSLD4mk3A3swUIc6Y2dclme?usp=sharing)
- [LLaMA efficient fine-tuning with LoRA](examples/llama/llama_lora.py)
- [LLaMA fine-tuning](examples/llama/llama.py)
- [GPT-J efficient fine-tuning with LoRA and INT8](examples/gptj/gptj_lora_int8.py) &ensp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hB_8s1V9K4IzifmlmN2AovGEJzTB1c7e?usp=sharing)
- [GPT-J efficient fine-tuning with LoRA](examples/gptj/gptj_lora.py)
- [GPT-2 efficient fine-tuning with LoRA](examples/gpt2/gpt2_lora.py) &ensp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Sh-ocNpKn9pS7jv6oBb_Q8DitFyj1avL/view?usp=sharing)


<br>

## 📊 Performance

Here is a comparison for the performance of different fine-tuning techniques on the LLaMA 7B model. We use the [Alpaca dataset](examples/llama/alpaca_data/) for fine-tuning. The dataset contains 52K instructions.

Hardware:

4xA100 40GB GPU, 335GB CPU RAM

Fine-tuning parameters:

```javascript
{
  'maximum sequence length': 512,
  'batch size': 1,
}
```

|      LLaMA 7B      | DeepSpeed + CPU Offloading | LoRA + DeepSpeed  | LoRA + DeepSpeed + CPU Offloading |
| --------- | ---- | ---- | ---- |
| GPU | 33.5 GB | 23.7 GB | 21.9 GB |
| CPU | 190 GB  | 10.2 GB | 14.9 GB |
| Time per epoch | 21 hours  | 20 mins | 20 mins |

Please submit your performance results on other GPUs.
<br >

## 📈 Roadmap
- [x] Support for LLaMA, GPT-J, GPT-2
- [ ] Dataset generation using self-instruction
- [ ] 2x more memory-efficient fine-tuning vs LoRA and unsupervised fine-tuning
- [ ] Evaluation of LLM models
- [ ] Support for Stable Diffusion

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
