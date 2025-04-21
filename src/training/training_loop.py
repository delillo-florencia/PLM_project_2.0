# --------------------- IMPORTS ---------------------

import os
import torch
import subprocess
import argparse
import yaml
import time
import shutil
import pytorch_lightning as pl
from utils.loss_functions import DistillationLoss
from lightning.data_module import ProteinDataModule
from utils.logging import get_latest_version
from training.pyl_training import ProteinTrainModule
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.distributed as dist
from pytorch_lightning.loggers import TensorBoardLogger
from utils.model_utils import int_env
from datetime import datetime
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # or DEBUG for more output
try: # check if FlashAttention can be used
    import flash_attn
    USE_FA = True
except ImportError:
    USE_FA = False


# --------------------- GLOBALS ---------------------

DEVICES = torch.cuda.device_count()
RANK = int_env("SLURM_PROCID", "RANK", default=0)
WORLD_SIZE = int_env("SLURM_NTASKS", "WORLD_SIZE", default=1)
LOCAL_WORLD_SIZE = int_env("LOCAL_WORLD_SIZE", default=torch.cuda.device_count())
LOCAL_RANK = int_env("SLURM_LOCALID", "LOCAL_RANK", default=0)
NODES = WORLD_SIZE // LOCAL_WORLD_SIZE
torch.set_float32_matmul_precision("high" if torch.cuda.is_bf16_supported() else "highest")
torch.cuda.set_device(LOCAL_RANK)

#RANK        = int_env("RANK",        "SLURM_PROCID", default=0)
#WORLD_SIZE  = int_env("WORLD_SIZE",  "SLURM_NTASKS", default=1)
#LOCAL_RANK  = int_env("LOCAL_RANK",  "SLURM_LOCALID", default=0)
#DEVICES     = torch.cuda.device_count()

#LOCAL_WORLD_SIZE = int_env("LOCAL_WORLD_SIZE", default=torch.cuda.device_count())
#NODES = WORLD_SIZE // LOCAL_WORLD_SIZE
#torch.set_float32_matmul_precision("high" if torch.cuda.is_bf16_supported() else "highest")
#torch.cuda.set_device(LOCAL_RANK)


# --------------------- PARAMETERS ---------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

output_dir = cfg["output_dir"]
run_name = cfg.get("run_name")
train_sampler_params = cfg["train_sampler_params"]
trainer_params = cfg["trainer_params"]
hashing_params = cfg["hashing_params"]
hyper_params = cfg["hyper_params"]



# ---------------------- TRAINING ----------------------

# unlimited batches scenario (using whole datalaoder)
if train_sampler_params["max_batch_num"] != -1:
    val_batch_num = int(train_sampler_params["max_batch_num"] * hashing_params["val_size_ratio"] / hashing_params["train_size_ratio"])
else:
    val_batch_num = -1

# infere fixed validation dataloader params
train_sampler_params["num_replicas"] = WORLD_SIZE
train_sampler_params["rank"] = RANK
val_sampler_params = train_sampler_params.copy()
val_sampler_params["max_batch_num"] = val_batch_num
    
# create output files
version_file = os.path.join(output_dir, run_name, ".version")
if int(os.getenv("RANK", 0)) == 0:
    run_name = datetime.today().strftime('%Y%m%d-%H%M%S') if run_name is None else run_name
    output_dir_ver, version = get_latest_version(output_dir, run_name)
    checkpoint_dir = os.path.join(output_dir_ver, 'checkpoints')
    os.makedirs(output_dir_ver, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(output_dir_ver, os.path.basename(args.config)))
    with open(version_file, "w") as f:
        f.write(version)
    print(f"@@@ RUN NAME: {run_name}_{version} @@@")
    print(f"Run Tensorbord: 'tensorboard --logdir={output_dir}'")
    print("Using mixed bf16-matmul precision." if torch.cuda.is_bf16_supported() else "Using true precision.")
    print("Flash attention will be used." if USE_FA else "No flash attention installed.")
    print("Max batches for val: ", val_sampler_params["max_batch_num"], ", test: ", train_sampler_params["max_batch_num"], sep="")
    print(f"OMP_NUM_THREADS set to {os.environ['OMP_NUM_THREADS']}. Using {int(os.environ['OMP_NUM_THREADS'])*WORLD_SIZE} cores out of {os.cpu_count()*NODES} in the system.")
else:
    version_file = os.path.join(output_dir, run_name, ".version")
    while not os.path.exists(version_file):
        time.sleep(0.1)
    version = open(version_file, "r").read().strip()
    output_dir_ver = os.path.join(output_dir, run_name, version)
    checkpoint_dir = os.path.join(output_dir_ver, 'checkpoints')

# turn on tensorboard
tb_logger = TensorBoardLogger(output_dir, name=run_name, version=version, default_hp_metric=False)

# get proper output dir and store hparams
tb_logger.log_hyperparams(hyper_params)

# load modules
train_data_module = ProteinDataModule(cfg["train_csv_file"], cfg["train_hash_file"], "train", train_sampler_params, collate_fn=lambda x: x)
val_data_module = ProteinDataModule(cfg["val_csv_file"], cfg["val_hash_file"], "val", val_sampler_params, collate_fn=lambda x: x)
model_module = ProteinTrainModule(**cfg["loop_params"], **hyper_params, use_saved_reps_logs_dir=cfg["precomputed_dir"], 
                                  output_dir=output_dir_ver, use_flash=USE_FA, distillation_loss=DistillationLoss())

# initialize trainer
trainer = pl.Trainer(
    **cfg["trainer_params"],
    logger=tb_logger,
    devices=DEVICES,
    num_nodes=NODES,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    enable_progress_bar=not args.disable_progress_bar,  
    enable_model_summary=True,
    use_distributed_sampler=False,
    precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 32,
    callbacks=[ModelCheckpoint(
        **cfg["checkpoint_params"], dirpath=checkpoint_dir, 
        filename="checkpoint_{epoch}_{step}")])

# run training
train_data_module.setup()
val_data_module.setup()
trainer.fit(model_module, train_dataloaders=train_data_module.dataloader(), val_dataloaders=val_data_module.dataloader())
