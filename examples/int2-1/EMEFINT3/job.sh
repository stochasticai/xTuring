#!/bin/bash

#SBATCH --job-name=QER-alpaca-ift
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 2-00:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 8              # Number of cores (-c)
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=120000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --ntasks=1
#SBATCH -o ./logs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=john.gkountouras@student.uva.nl

module load 2022
module load Anaconda3/2022.05

eval "$(conda shell.bash hook)"
conda activate gptq

python finetune.py --intq_checkpoint /home/gkounto/GPTQ4LLAMA/llama7b-2bit-128g.pt --wbits 2 --output_dir /scratch-shared/gkounto/lora-alpaca-longertrain-noinputs --micro_batch_size 64 --batch_size 128 --peft_pretrained /scratch-shared/gkounto/tmp/models/2bitqer32_toift --num_epochs 3 --train_on_inputs False --wandb_run_name alpaca2bitqerlora-int2-bs48-3epochs-notrainoninputs
