#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --job-name=get_reps_token_size
#SBATCH --output=reps_output_tokens.txt
#SBATCH --gres=gpu:4              
#SBATCH --mem=600G                
#SBATCH --partition=compute


torchrun --standalone --nnodes=1 --nproc_per_node=4 src/training/get_reps_token_size.py



hostname

