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

export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python /home/dtuteam/workspace/PLM_project_2.0/src/training/training_loop.py \
              --config /home/dtuteam/workspace/PLM_project_2.0/src/configs/config.yaml