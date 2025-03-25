#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --job-name=uniref100_job
#SBATCH --output=uniref100_output.txt

bash download.sh

hostname