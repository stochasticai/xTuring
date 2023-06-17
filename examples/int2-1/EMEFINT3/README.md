# Extremely Memory-Efficient Fine-tuning of Large Language Models

This part repository is the official implementation of EMEF. It is basically a fork of [PEFT](https://github.com/huggingface/peft)
with extra functionalities.

## Installation


```
conda activate LRECINT3
python setup.py install
```

## GPTQ models
To create the GPTQ LLaMA models, follow the instructions in the [GPTQ-for-LLaMA](
https://github.com/qwopqwop200/GPTQ-for-LLaMa) repository. The parameters for generating
the GPTQ models are available in Appendix A of our paper.

## Training

To run the EMEF pipeline for the model(s) as described in the paper, run this command:

```
python finetune.py
```

The arguments are:
- `base_model`: The base model for training. Default is 'decapoda-research/llama-7b-hf'.
- `intq_checkpoint`: The intq checkpoint. Default is 'llama7b-2bit-128g.pt'.
- `wbits`: The number of bits for weight quantization. Default is 4.
- `groupsize`: The group size. Default is 128.
- `data_path`: The path to the data. Default is 'yahma/alpaca-cleaned'.
- `output_dir`: The output directory for the trained model. Default is './lora-alpaca'.
- `batch_size`: The batch size for training. Default is 144.
- `micro_batch_size`: The size of the micro-batches. Default is 1.
- `num_epochs`: The number of epochs for training. Default is 3.
- `learning_rate`: The learning rate. Default is 3e-4.
- `cutoff_len`: The cutoff length for training. Default is 512.
- `val_set_size`: The size of the validation set. Default is 2000.
- `lora_r`: The Lora r value. Default is 8.
- `lora_alpha`: The Lora alpha value. Default is 16.
- `lora_dropout`: The Lora dropout rate. Default is 0.05.
- `lora_target_modules`: List of target modules for Lora. Default is q_proj, v_proj.
- `train_on_inputs`: Whether to train on inputs. If False, masks out inputs in loss. Default is True.
- `group_by_length`: If True, groups sequences by length. It's faster but may produce an odd training loss curve. Default is False.
- `wandb_project`: The name of the Weights & Biases project for experiment tracking.
- `wandb_run_name`: The name of the Weights & Biases run for experiment tracking.
- `wandb_watch`: What to watch in Weights & Biases. Options are 'false', 'gradients', 'all'. Default is 'all'.
- `wandb_log_model`: Whether to log the model to Weights & Biases. Options are 'false', 'true'.
- `resume_from_checkpoint`: Path to resume training from a checkpoint or final adapter. Default is None.
- `int8`: Whether to use 8-bit integers for quantization. Default is False.
- `peft_pretrained`: Path to the pretrained Peft model. Default is ''.


## Pre-trained Models

You can download pretrained models here:

- [Dropbox](https://www.dropbox.com/sh/men4rrxs4slhq4r/AAApp_4lcD9uXJKn31Zb1A-Ea?dl=0)

Files:

- `13bint2lrec`: 13B INT2 LREC error-corrected model
- `13bint4lrec`: 13B INT4 LREC error-corrected model
- `int2lrec`: 7B INT2 LREC error-corrected model
- `int3lrec`: 7B INT3 LREC error-corrected model
- `int4lrec`: 7B INT4 LREC error-corrected model
- `lora-alpaca-int4-emef`: 7B INT4 EMEF model
- `int2lrec-emef`: 7B INT2 LREC error-corrected EMEF model

The uploader's name was randomly generated to adhere to anonymity requirements.