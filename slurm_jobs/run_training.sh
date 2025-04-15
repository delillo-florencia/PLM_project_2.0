#!/bin/bash
#SBATCH --ntasks-per-node=1      # 1 task per node (for launcher)
#SBATCH --cpus-per-task=4        # 4 CPUs per task
#SBATCH --job-name=plm_train
#SBATCH --output=training_output.log
#SBATCH --gres=gpu:1             # 1 GPU per node
#SBATCH --mem=30G
#SBATCH --partition=g2
#SBATCH --error=training_error.log

# --- Critical fixes below ---

# 1. Use Slurm's node list instead of hardcoding --nnodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12345  # Fixed port for communication
# Set NCCL environment variables for better fault tolerance
export NCCL_BLOCKING_WAIT=1          # Make NCCL operations blocking (wait until completion)
export NCCL_ASYNC_ERROR_HANDLING=1   # Enable async error handling (abort on errors)
export NCCL_SOCKET_TIMEOUT_MS=60000  # Increase socket timeout to 60 seconds (default is much lower)

# 2. Use srun instead of direct torchrun for better Slurm integration
srun --ntasks=$SLURM_JOB_NUM_NODES \
     --nodes=$SLURM_JOB_NUM_NODES \
     torchrun --nnodes=$SLURM_JOB_NUM_NODES \
              --nproc_per_node=1 \
              --rdzv_id=$SLURM_JOB_ID \
              --rdzv_backend=c10d \
              --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
              /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
              --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml

