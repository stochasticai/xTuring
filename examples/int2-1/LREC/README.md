# Low-Rank Error Correction of Large Language Models

This part repository is the official implementation of LREC. 

## Installation


```
conda create --name LREC python=3.9 -y
conda activate LREC
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## GPTQ Models
To create the GPTQ LLaMA models, follow the instructions in the [GPTQ-for-LLaMA](
https://github.com/qwopqwop200/GPTQ-for-LLaMa) repository. The parameters for generating
the GPTQ models are available in Appendix A of our paper.

## Caching Teacher Outputs
To cache the teacher outputs, run the following command:

```
python cachedistillationoutputs.py --base_model <base_model> --seqlen <seqlen> --n_samples <n_samples> --train_cache_dir <train_cache_dir> --val_cache_dir <val_cache_dir> --seed <seed>
```

The arguments are:
- `base_model`: the base model for caching (default is `decapoda-research/llama-7b-hf`)
- `seqlen`: the sequence length (default is 2048)
- `n_samples`: number of samples (default is 10000)
- `train_cache_dir`: directory for training cache (you need to replace `###ANONYMIZED###` with your specific path)
- `val_cache_dir`: directory for validation cache (you need to replace `###ANONYMIZED###` with your specific path)
- `seed`: the random seed (default is 1)

Do note that depending on the number of outputs you are caching, this may use a lot of disk space. For example, caching the outputs of one teacher model on 10000 samples requires 1.2 TB of disk space.

## Training

To run the error-correction pipeline for the model(s) as described in the paper, run this command:

```
python lrec.py --base_model <base_model> --intq_checkpoint <intq_checkpoint> --wbits <wbits> --groupsize <groupsize> --lora_alpha <lora_alpha> --lora_r <lora_r> --lora_dropout <lora_dropout> --lora_target_modules <lora_target_modules> --n_samples <n_samples> --lr <lr> --batch_size <batch_size> --num_epochs <num_epochs> --kl_weight <kl_weight> --ce_weight <ce_weight> --trainable_kl_weight <trainable_kl_weight> --trainable_ce_weight <trainable_ce_weight> --weight_decay <weight_decay> --save_freq <save_freq> --intra_save_freq <intra_save_freq> --seed <seed> --seqlen <seqlen> --cache <cache> --train_cache_dir <train_cache_dir> --val_cache_dir <val_cache_dir> --ckpt_dir <ckpt_dir>
```

The arguments are:

- `base_model`: The base model for training. Default is 'decapoda-research/llama-7b-hf'.
- `intq_checkpoint`: The intq checkpoint. Default is 'llama7b-2bit-128g.pt'.
- `wbits`: The number of bits for weight quantization. Default is 2.
- `groupsize`: The group size. Default is 128.
- `lora_alpha`: The Lora alpha value. Default is 128.
- `lora_r`: The Lora r value. Default is 32.
- `lora_dropout`: The Lora dropout rate. Default is 0.05.
- `lora_target_modules`: List of target modules for Lora. Default is q_proj, v_proj, k_proj, o_proj, up_proj, down_proj, gate_proj.
- `n_samples`: The number of samples. Default is 2048.
- `lr`: The learning rate. Default is 3e-4.
- `batch_size`: The batch size. Default is 4.
- `num_epochs`: The number of epochs. Default is 20.
- `kl_weight`: The KL weight. Default is 1.
- `ce_weight`: The CE weight. Default is 200.
- `trainable_kl_weight`: Whether to learn the KL weight. Default is False.
- `trainable_ce_weight`: Whether to learn the CE weight. Default is False.
- `weight_decay`: The weight decay. Default is 1e-5.
- `save_freq`: The frequency (period) of saving checkpoints in epochs. Default is 1.
- `intra_save_freq`: The period (in num_batches) of saving checkpoints within an epoch. Default is 200.
- `seed`: The random seed. Default is 0.
- `seqlen`: The sequence length. Default is 2048.
- `cache`: Whether to use cached distillation outputs. Default is True.
- `train_cache_dir`: Training cache directory. Replace the '###ANONYMIZED###' placeholder with your actual path.
- `val_cache_dir`: Validation cache directory. Replace the '###ANONYMIZED###' placeholder with your actual path.
- `ckpt_dir`: The directory for saving and loading checkpoints. Replace the placeholder with your actual path if needed.
- `save_dir`: The directory for saving the final model. Replace the placeholder with your actual path if needed.


## Evaluation

To evaluate the perplexities and KL divergences from our model to a reference model, run:

```
python eval.py python eval.py --mode <mode> --base_model <base_model> --intq_checkpoint <intq_checkpoint> --wbits <wbits> --groupsize <groupsize> --peft_pretrained <peft_pretrained> --reference_model <reference_model> --double_lora <double_lora> --stride <stride> --dataset <dataset> --cutoff_len <cutoff_len> --col <col>
```
- `mode`: Choose the mode for the script, options are 'perplexity', 'alpaca_perplexity', 'kl_divergence'.
- `base_model`: The base model for evaluation. Default is 'decapoda-research/llama-7b-hf'.
- `intq_checkpoint`: The intq checkpoint for evaluation. Default is 'llama7b-2bit-128g.pt'.
- `wbits`: The number of bits for weight quantization. Default is 3.
- `groupsize`: The group size for evaluation. Default is 128.
- `peft_pretrained`: Path to pretrained Peft model.
- `reference_model`: Path to reference model.
- `double_lora`: Path to second Peft model (optional).
- `stride`: Stride length. Default is 512.
- `dataset`: The dataset for evaluation. Default is 'alpaca'.
- `cutoff_len`: Cutoff length. Default is 512.
- `col`: Column name for the dataset. Default is 'sentence'.


## C4 Evaluation
To evaluate on the C4 dataset we use a different script that caches the tokenized dataset
to disk in order to speed up later runs. To execute this script, run:

```
python c4eval.py --base_model <base_model> --intq_checkpoint <intq_checkpoint> --wbits <wbits> --groupsize <groupsize> --peft_pretrained <peft_pretrained> --stride <stride> --encodings_pickle <encodings_pickle> --batch_size <batch_size>
```
- `base_model`: The base model for C4 evaluation. Default is 'decapoda-research/llama-7b-hf'.
- `intq_checkpoint`: The intq checkpoint for C4 evaluation. Default is 'llama7b-2bit-128g.pt'.
- `wbits`: The number of bits for weight quantization. Default is 4.
- `groupsize`: The group size for C4 evaluation. Default is 128.
- `peft_pretrained`: Path to pretrained Peft model for C4 evaluation.
- `stride`: Stride length. Default is 2048.
- `encodings_pickle`: Path to the pickle file containing the tokenized C4 dataset. Replace the '###ANONYMIZED###' placeholder with your actual path.
- `batch_size`: Batch size for C4 evaluation. Default is 8.

This is a long process, it takes about 10 hours to run on a single A100 GPU. We also use a stride of `2048` instead of `512` to speed up the process.


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

## Slurm Scripts

We provide [Slurm scripts](slurm_helper_scripts/) for easier training and evaluation on a cluster.