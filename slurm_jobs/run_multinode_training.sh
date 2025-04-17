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

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=29500

# launch one torchrun_slurm on each node via srun
srun torchrun_slurm \
     --rdzv_backend=c10d \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
     --rdzv_id=${SLURM_JOB_ID} \
     --rdzv_configs="join_timeout=1800,last_call_timeout=60" \
     /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
              --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml