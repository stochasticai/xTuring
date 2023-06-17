#!/bin/bash

#SBATCH --job-name=LREC
#SBATCH -p gpu  # Partition to submit to
#SBATCH -t 1-12:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 8              # Number of cores (-c)
#SBATCH --array=1-20
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=64000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --ntasks=1
#SBATCH -o ./logs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=###ANONYMIZED###

module load ###ANONYMIZED###
module load ###ANONYMIZED###

eval "$(conda shell.bash hook)"
conda activate gptq

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" args.txt)

python lrec.py $ARGS
