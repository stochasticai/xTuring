# INT2.1 Efficient Large Language Model Framework

This part repository is the official implementation of "INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation". We provide the code for regenerating our results and using our INT2.1 framework. 

<br>

## 💡 Key Contributions

Our INT2.1 framework not only greatly reduces the VRAM requirement for fine-tuning, but also significantly improves quantized LLMs’ performance. It holds promising implications for the future development and optimization of quantized models, marking a pivotal shift in the landscape of low-resource machine learning computations. Key contributions of this framework are: 

- An Extremely Memory-Efficient Finetuning (EMEF) method that integrates low-rank adaptation, reducing memory requirement by 5.6x and enabling fine-tuning of LLMs on lower-resource computing devices, such as a consumer-grade laptop. 
- A quantization-agnostic error correction method, Low-Rank Error Correction (LREC), that readily generalizes to any quantization standards, such as INT2, INT3, and INT4, restoring their lost performance due to quantization.
- A fully functional INT2 Large Language Model that is capable of generating coherent human-level text, outperforming models compressed using prior techniques.

<br>

## ⚙️ Environment Setup

Our INT2 and INT4 implementations use the same code. The INT3 implementation requires a separate environment, as it uses a different version of EMEF.
When running experiments, make sure to activate the correct environment depending on the required precision.

To run the INT2 and INT4 experiments, use the following `LREC` environment:
```
conda create --name LREC python=3.9 -y
conda activate LREC
cd EMEF
python setup.py install
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
cd ../LREC
pip install -r requirements.txt
cd ..
```
To run the INT3 experiments, use the following `LRECINT3` environment:
```
conda create --name LRECINT3 python=3.9 -y
conda activate LRECINT3
cd EMEFINT3
python setup.py install
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
cd ../LREC
pip install -r requirements.txt
```
<br>

## 📚 Framework Manuals

Our Extremely Memory-Efficient Fine-tuning (EMEF) method and  Low-Rank Error Correction (LREC) method are implemented in separate folders. 
To use our framework, refer to the README.md manuals in the respective folders: [EMEF](EMEF/README.md) and [LREC](LREC/README.md).

<br>

## 📄 Citation
If you find INT2.1 Framework useful or relevant to your project, please kindly cite our paper:
```
@misc{2023int21,
    title={INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation}, 
    author={Yuji Chai and John Gkountouras and Glenn G. Ko and David Brooks and Gu-Yeon Wei},
    year={2023},
    eprint={2306.08162},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<br>

## 🤝 Acknowledgement
Our implementation is inspired by:
- [alpaca-lora](https://github.com/tloen/alpaca-lora)
- [GPTQ-for-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
- [peft](https://github.com/huggingface/peft)

Special thanks to their contributions to the community.

You can also join our [Discord server](https://discord.gg/TgHXuSJEk6) and start a discussion in the `#xturing` channel.

<br>

## 🌎 Contributing
As an open source project in a rapidly evolving field, we welcome contributions of all kinds, including new features and better documentation. Please read our [contribution guide](../../CONTRIBUTING.md) to learn how you can get involved.