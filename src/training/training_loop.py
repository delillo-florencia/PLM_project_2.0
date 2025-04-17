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
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # or DEBUG for more output
try: # check if FlashAttention can be used
    import flash_attn
    USE_FA = True
except ImportError:
    USE_FA = False



# --------------------- GLOBALS ---------------------

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
#LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
DEVICES = torch.cuda.device_count()
torch.set_float32_matmul_precision("high" if torch.cuda.is_bf16_supported() else "highest")
torch.cuda.set_device(RANK)



# --------------------- PARAMETERS ---------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
parser.add_argument("--progress_bar", type=bool, default=True, help="Enable progress bar (True/False)")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

csv_file = cfg["csv_file"]
train_hash_file = cfg["train_hash_file"]
val_hash_file = cfg["val_hash_file"]
precomputed_dir = cfg["precomputed_dir"]

output_dir = cfg["output_dir"]
run_name = cfg.get("run_name")
loop_params = cfg["loop_params"]
train_sampler_params = cfg["train_sampler_params"]

checkpoint_params = cfg["checkpoint_params"]
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
val_sampler_params = {
    "max_batch_tokens": train_sampler_params["max_batch_tokens"],
    "shuffle": False, "shuffle_batch_order": False,
    "max_batch_num": val_batch_num}
val_sampler_params["num_replicas"] = DEVICES
val_sampler_params["rank"] = RANK
train_sampler_params["num_replicas"] = DEVICES
train_sampler_params["rank"] = RANK
print("Creating output files..")
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
    print("================")
    print(f"Run Tensorbord: 'tensorboard --logdir={output_dir}'")
    print("Using mixed bf16-matmul precision." if torch.cuda.is_bf16_supported() else "Using true precision.")
    print("Flash attention will be used." if USE_FA else "No flash attention installed.")
    print("Max batches for val: ", val_sampler_params["max_batch_num"], ", test: ", train_sampler_params["max_batch_num"], " ,devices: ", DEVICES, sep="")
    print("================")
else:
    while not os.path.exists(version_file):
        time.sleep(0.1)
    output_dir_ver = os.path.dirname(version_file)
    version = open(version_file).read().strip()
    checkpoint_dir = os.path.join(output_dir_ver, 'checkpoints')

print("TensorLoger")
# turn on tensorboard for convenience
tb_logger = TensorBoardLogger(output_dir, name=run_name, version=version, default_hp_metric=True)

# get proper output dir and store hparams
tb_logger.log_hyperparams(hyper_params)

# load modules
print("Loading_models")
train_data_module = ProteinDataModule(csv_file, train_hash_file, train_sampler_params, collate_fn=lambda x: x)
val_data_module = ProteinDataModule(csv_file, val_hash_file, val_sampler_params, collate_fn=lambda x: x)
model_module = ProteinTrainModule(**loop_params, **hyper_params, use_saved_reps_logs_dir=precomputed_dir, 
                                  output_dir=output_dir_ver, use_flash=USE_FA, distillation_loss=DistillationLoss())
print("Initializing trainer")
# initialize trainer
trainer = pl.Trainer(
    **trainer_params,
    logger=tb_logger,
    devices=DEVICES,

    num_nodes=WORLD_SIZE,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    enable_progress_bar=args.progress_bar,  
    enable_model_summary=True,
    use_distributed_sampler=False,
    precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 32,
    callbacks=[ModelCheckpoint(
        **checkpoint_params, dirpath=checkpoint_dir, 
        filename="checkpoint_{epoch}_{step}")])

# run training
print("Running_training")
train_data_module.setup()
val_data_module.setup()
trainer.fit(model_module, train_dataloaders=train_data_module.dataloader(), val_dataloaders=val_data_module.dataloader())
