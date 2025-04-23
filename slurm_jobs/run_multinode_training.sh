#!/bin/bash
#SBATCH --job-name=plm_train
#SBATCH --partition=g2
#SBATCH --nodes=20                   
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --output=training_output.log
#SBATCH --error=training_error.log

#SBATCH --exclude=hpccluster-g2-ghpc-[67,69,27,45,52,26,0]

MASTER_PORT=29540

# 1) Extract the real head node name
read -r head_node <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
# 2) Extract the first external IP from that node
read -r head_ip <<< "$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')"

# Launch
srun torchrun \
     --nnodes=$SLURM_JOB_NUM_NODES \
     --nproc_per_node=1 \
     --rdzv_backend=c10d \
     --rdzv_endpoint=${head_ip}:${MASTER_PORT} \
     --rdzv_id=${SLURM_JOB_ID} \
     --rdzv_conf timeout=3600,heartbeat=1800 \
     /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
          --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml \
          --num_workers=3
     
