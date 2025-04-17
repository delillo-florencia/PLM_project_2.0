#!/bin/bash
#SBATCH --job-name=plm_train
#SBATCH --partition=g2
#SBATCH --nodes=2                   
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --output=training_output.log
#SBATCH --error=training_error.log

export MASTER_PORT=29500

# launch one torchrun_slurm on each node via srun
srun torchrun_slurm \
     --rdzv_backend=c10d \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_endpoint=${SLURM_JOB_NODELIST}:${MASTER_PORT} \
     /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
              --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml