#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=45:00:00
#SBATCH --job-name=training_loop
#SBATCH --output=training_output.txt
#SBATCH --gres=gpu:4              
#SBATCH --mem=600G                
#SBATCH --partition=compute


torchrun --standalone --nnodes=1 --nproc_per_node=4 src/training/training.py



hostname                                                                                   
~                       
