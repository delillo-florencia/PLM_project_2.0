#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --job-name=get_reps_test
#SBATCH --output=reps_output.txt

python src/training/get_reps.py

hostname
