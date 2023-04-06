<h1 align="center">Fine-tune LLMs with only 6GB of Memory</h1>

The recent progress in large language model such as ChatGPT, GPT-4 has intrigued unprecedented interest in
large language models. Centralized solution such as `OpenAI API` has made building around an existing model easy. But, for developers who want a model that is trained on their own data and tailored to their own vertical domain of applications, there is still a great entry barrier to overcome. Even with the help of parameter efficient fine-tuning methods such as LoRA and INT8-based training library, such as the `bitsandbyptes`, it is only realistic to fine-tune any LLM with 7B+ parameter on a latest GPU with the topnotch specifics, such as a Nvidia RTX4090, as demonstrated by `alpaca-lora`. As a result, it is still a norm in LLM community, that you have to pay a premium price for your hardware before even thinking of building your own model with your own data.

However, with recent advances in extreme compression methods, such as GPTQ, fine-tuning your own model with common place consumer GPU has become a reality. With our implementation, which combines the power of GPTQ and LoRA, the memory requirement for fine-tuning (fine-tuning a llama-like 7B parameter model on a dataset with input sequence length of 512) has been reduced to lower than 6GB, which means 8 out of the top 10 most popular GPU on [steam hardware survey](https://store.steampowered.com/hwsurvey/videocard/) (which accounts for nearly 50% of all steam users with GPU) already have sufficient memory for fine-tuning. We believe this method could open the door of LLM fine-tuning to a much wider community and further democratize the power of LLM by drastically lowering the hardware requirement.

In the following sections, we will discuss its runtime performance (such as memory consumption and training time) and a tutorial on how to apply this method with `xturing`.

<br>

## Advantages

With INT4 fine-tuning, xTuring offers several benefits:

1. Reduced memory footprint: The INT4 precision reduces the memory requirements to just ~6GB of VRAM, allowing you to fine-tune LLMs on hardware with lower memory capacities.
2. Cheaper fine-tuning: The reduced memory requirements and INT4 precision enable cheaper fine-tuning of LLMs by using less powerful resources.
3. LoRA support: This update includes support for the LLaMA LoRA model, a powerful architecture that enables efficient fine-tuning and high-quality results.
4. Data privacy and security: The entire fine-tuning process can be carried out on your local computer or private cloud, ensuring the confidentiality and security of your data.
5. WandB integration: Easily monitor the fine-tuning process with integrated WandB logging, allowing you to track progress.

It is a great step on the way to democratize large language models and making them accessible to everyone.


## 📊 Performance

Here is a comparison for the performance of different fine-tuning techniques on the LLaMA 7B model. We use the [Alpaca dataset](examples/llama/alpaca_data/) for fine-tuning. The dataset contains 52K instructions.

### > Fine-tuning parameters:

```javascript
{
  'maximum sequence length': 512,
  'batch sizes': [1, 24, 48],
}
```
### > Hardware:

Here are our benchmark results collected from common customer grade GPU and server grade GPU. Explanations for abbreviations: _`OOM`_ stands for Out-Of-Memory, which means the hardware does not have enough memory to support running a certain training configuration and _`BS`_ stands for the micro batch size, which indicates how many training examples are pushed onto device for every training step.

**- 1x RTX3070 8GB Laptop GPU**

|      LLaMA 7B      | FP16 + LoRA | INT8 + LoRA  | INT4 + LoRA (Ours) |
| --- | --- | --- | --- |
| GPU Memory (GB) @ _`BS`_=1 | _`OOM`_ | _`OOM`_ |  **4.93** |
| Time per Epoch (min) @ _`BS`_=1 | _`OOM`_ | _`OOM`_ | **427*** |

**- 1x A100 40GB GPU**

|      LLaMA 7B      | FP16 + LoRA | INT8 + LoRA  | INT4 + LoRA (Ours) |
| --- | --- | --- | --- |
| GPU Memory (GB) @ _`BS`_=1 |  14.6 |  8.85 |  **5.96** |
| Time per Epoch (min) @ _`BS`_=1 |   220* |  643* | **355*** |
| GPU Memory (GB) @ _`BS`_=24 |  24.08 |  20.4 |  **17.1** |
| Time per Epoch (min) @ _`BS`_=24 | 79.1 | 168*  | **240*** |
| GPU Memory (GB) @ _`BS`_=48 | _`OOM`_ |  32.78 |  **27.32** |
| Time per Epoch (min) @ _`BS`_=48 | _`OOM`_ |   110.6 |   **81.8** |

*: Extrapolated estimates, we stopped the training early due to time constraints.

More benchmark results and performance analysis on different hardware and batch sizes will be posted. You are encouraged to submit your performance results on other GPUs/configs/models to help us improve this table.

<br>

## 📚 Tutorial

All instructions are inside the example [notebook](LLaMA_lora_int4.ipynb)

## 🤝 Acknowledgement
Our implementation is inspired by:
- alpaca-lora
- GPTQ4LLama
- peft

Special thanks to their contributions to the community.

You can also join our [Discord server](https://discord.gg/TgHXuSJEk6) and start a discussion in the `#xturing` channel.

<br>

## 📝 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

<br>

## 🌎 Contributing
As an open source project in a rapidly evolving field, we welcome contributions of all kinds, including new features and better documentation. Please read our [contributing guide](CONTRIBUTING.md) to learn how you can get involved.
