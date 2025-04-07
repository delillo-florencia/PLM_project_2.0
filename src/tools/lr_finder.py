#!/usr/bin/env python
import os
import argparse
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import time
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import TensorBoardLogger

from lightning.data_module import ProteinDataModule
from training.pyl_training import ProteinTrainModule
from utils.loss_functions import DistillationLoss
from utils.logging import get_latest_version

try:
    import flash_attn
    USE_FA = True
except ImportError:
    USE_FA = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learning Rate Finder using Lightning's Tuner API"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration from YAML
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Get configuration parameters (adjust keys as necessary)
    csv_file = cfg["csv_file"]
    train_hash_file = cfg["train_hash_file"]
    train_sampler_params = cfg["train_sampler_params"]
    trainer_params = cfg["trainer_params"]
    loop_params = cfg["loop_params"]
    hyper_params = cfg["hyper_params"]
    precomputed_dir = cfg.get("precomputed_dir", None)
    output_dir = cfg["output_dir"]

    # For LR finder, run on one device
    DEVICES = torch.cuda.device_count()
    RANK = int(os.environ.get("SLURM_PROCID", 0))
    train_sampler_params["num_replicas"] = DEVICES
    train_sampler_params["rank"] = RANK

    # Create a unique run name and output directory
    version_file = os.path.join(output_dir, cfg["run_name"], ".version")
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        output_dir_ver, version = get_latest_version(output_dir, cfg["run_name"])
        os.makedirs(output_dir_ver, exist_ok=True)
        shutil.copy(args.config, os.path.join(output_dir_ver, os.path.basename(args.config)))
        with open(version_file, "w") as f:
            f.write(version)
    else:
        while not os.path.exists(version_file):
            time.sleep(0.1)
        output_dir_ver = os.path.dirname(version_file)
        version = open(version_file).read().strip()
        checkpoint_dir = os.path.join(output_dir_ver, 'checkpoints')

    # Set up a TensorBoard logger (optional but useful)
    tb_logger = TensorBoardLogger(save_dir=output_dir, name=cfg["run_name"], version=version, default_hp_metric=True)

    # Set up your DataModule (only training is needed for the LR finder)
    train_data_module = ProteinDataModule(csv_file, train_hash_file, train_sampler_params, collate_fn=lambda x: x)
    train_data_module.setup()  # Prepares the dataloader
    train_dataloader = train_data_module.dataloader()

    # Instantiate your model module with the required arguments
    model_module = ProteinTrainModule(
        **loop_params, 
        **hyper_params, 
        use_saved_reps_logs_dir=precomputed_dir, 
        output_dir=output_dir_ver,
        use_flash=USE_FA,
        distillation_loss=DistillationLoss()
    )

    # Create a Trainer (1 GPU/CPU is enough for LR finder)
    trainer = pl.Trainer(
        **trainer_params,
        logger=tb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true",
        devices=DEVICES,
        use_distributed_sampler=False,
    )

    # Create a Tuner instance and run the LR finder with specified range
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model_module,
        train_dataloaders=train_dataloader,
        max_lr=10.0,
        min_lr=1e-10,
    )

    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested Learning Rate: {suggested_lr}")
        fig = lr_finder.plot(suggest=True)
        plot_path = os.path.join(output_dir_ver, "lr_finder.png")
        fig.savefig(plot_path)
        print(f"Saved LR finder plot to {plot_path}")

if __name__ == "__main__":
    main()
