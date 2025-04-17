#!/bin/bash
#SBATCH --job-name=plm_train
#SBATCH --partition=g2
#SBATCH --nodes=4                   
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --output=training_output.log
#SBATCH --error=training_error.log


MASTER_PORT=29200

# 1) Extract the real head node name
read -r head_node <<< "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)"
# 2) Resolve it to an IP that all nodes can actually connect to
read -r head_ip <<< "$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)"

# launch one torchrun_slurm on each node via srun
srun torchrun_slurm \
     --rdzv_backend=c10d \
     --rdzv_id=$SLURM_JOB_ID \
     --rdzv_endpoint=${head_ip}:${MASTER_PORT} \
     --rdzv_id=${SLURM_JOB_ID} \
     /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
              --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml
              