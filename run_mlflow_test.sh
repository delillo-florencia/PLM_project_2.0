#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --job-name=test_mlflow
#SBATCH --output=mlflow.txt
#SBATCH --gres=gpu:1        
#SBATCH --mem=600G                
#SBATCH --partition=compute

# Set the MLFLOW_TRACKING_URI to point to Google Cloud Storage bucket
export MLFLOW_TRACKING_URI="http://34.13.195.239:5000"

python  src/training/mlflow_test.py



hostname
