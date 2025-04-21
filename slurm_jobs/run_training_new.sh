#!/bin/bash
#SBATCH --nodes=2               # Request 2 nodes
#SBATCH --ntasks-per-node=1      # 1 task per node (for launcher)
#SBATCH --cpus-per-task=4        # 4 CPUs per task
#SBATCH --job-name=plm_train
#SBATCH --output=training_output.log
#SBATCH --gres=gpu:1             # 1 GPU per node
#SBATCH --mem=30G
#SBATCH --partition=g2
#SBATCH --exclusive  
#SBATCH --error=training_error.log
#SBATCH --time=24:00:00          # Add time limit to prevent hanging

# --- Critical fixes below ---

# 1. Properly set up the master address and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# 2. Set NCCL environment variables for better reliability
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
#export NCCL_IB_DISABLE=1          # Disable Infiniband if not using it
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_TIMEOUT_MS=60000
export PYTHONFAULTHANDLER=1

# 3. Use torchrun directly with proper parameters
# Calculate the number of processes per node (should be 1 in your case)
PROC_PER_NODE=1

# Launch the training script
srun --nodes 2 -l /bin/hostname torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$PROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
    --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml
