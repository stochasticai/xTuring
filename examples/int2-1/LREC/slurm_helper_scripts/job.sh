#!/bin/bash

#SBATCH --job-name=LREC
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 2-00:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 8              # Number of cores (-c)
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=120000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --ntasks=1
#SBATCH -o ./logs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=###ANONYMIZED###

module load ###ANONYMIZED###
module load ###ANONYMIZED###

eval "$(conda shell.bash hook)"
conda activate gptq

python lrec.py --base_model decapoda-research/llama-13b-hf --intq_checkpoint llama13b-2bit-128g.pt --wbits 2 --ce_weight 120 --kl_weight 0.5 --lora_target_modules q_proj v_proj k_proj o_proj up_proj gate_proj --n_samples 10000 --batch_size 6 --lr 1e-5 --num_epochs 10 --save_freq 1 --cache --intra_save_freq 400 --train_cache_dir "###ANONYMIZED###" --val_cache_dir "###ANONYMIZED###


