#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --job-name=hashing
#SBATCH --output=hashing.txt
#SBATCH --mem=30G
#SBATCH --partition=g2


python src/tools/run_hashing.py data/final_uniref100.csv data/uniref100




hostname

