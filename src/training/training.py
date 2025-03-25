# --------------------- IMPORTS ---------------------

import os
import torch
import subprocess
import pytorch_lightning as pl
from utils.loss_functions import DistillationLoss
from lightning.data_module import ProteinDataModule
from utils.logging import get_latest_version
from training.pyl_training import SetEpochCallback, ProteinTrainModule
from lightning.callbacks import ModelCheckpoint
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
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
DEVICES = torch.cuda.device_count()
torch.set_float32_matmul_precision("high" if torch.cuda.is_bf16_supported() else "highest")
print("Using mixed bf16-matmul precision." if torch.cuda.is_bf16_supported() else "Using true precision.")
print("Flash attention will be used." if USE_FA else "No flash attention installed.")
torch.cuda.set_device(LOCAL_RANK)



# --------------------- PARAMETERS ---------------------

# define input data sources
csv_file = "/home/developer/Projects/PLM_project_2.0/data/uniprot_data_500k_sampled.csv"
train_hash_file = "/home/developer/Projects/PLM_project_2.0/data/uniprot_data_500k_sampled_train.hash"
val_hash_file = "/home/developer/Projects/PLM_project_2.0/data/uniprot_data_500k_sampled_val.hash"
precomputed_dir = "/home/developer/Projects/PLM_project_2.0/data/outputs/timestamp" # used for offline training

# define run name and master output dir!
output_dir = "/home/developer/Projects/PLM_project_2.0/outputs/"
run_name = "test2"

# training parameters
loop_params = {
    "student_model_param": "8M", # student model
    "teacher_model_param": "35M", # teacher model
    "save_masked_sequences": False, # should masked seqeunces be saved
    # hyperparameters
    "learning_rate": 1e-4,
    # online or offline destillation?
    "use_saved_reps_logs": False, # should we use precomputed reps and logits?
    # training metrics saving modes
    "save_reps_logs": False, # save representations and logits per batch (only first epoch)
    "save_per_batch": False, # save metrics per batch, by default its saved only per epoch
    }

# dataloader parameters
validation_size_ratio = 0.2
train_sampler_params = {
    "max_batch_tokens": 2000, # maximum token number per batch
    "shuffle": False, # all samples before bucketing
    "shuffle_batch_order": True, # batch order after bucketing
    "max_batch_num": 100, # max number of batches across all GPUs
    } 

# checkpoint parameters
checkpoint_params = {
    "every_n_epochs": 25,  # save per epoch (use 'every_n_train_steps' for per batch)
    "save_top_k": 1, # -1 if save all, or 1 to keep only the last one
    }

# extra trainer settings
trainer_params = {
    "max_epochs": 3 # set epoch number
    }


# ---------------------- TRAINING ----------------------
#                 @ DO NOT MODIFY HERE @

# infere fixed validation dataloader params
val_sampler_params = {
    "max_batch_tokens": train_sampler_params["max_batch_tokens"],
    "shuffle": False, "shuffle_batch_order": False,
    "max_batch_num": int(train_sampler_params["max_batch_num"]*validation_size_ratio)}
print("Max batches for val: ", val_sampler_params["max_batch_num"], ", test: ", train_sampler_params["max_batch_num"], " ,devices: ", DEVICES, sep="")
val_sampler_params["num_replicas"] = DEVICES
val_sampler_params["rank"] = RANK
train_sampler_params["num_replicas"] = DEVICES
train_sampler_params["rank"] = RANK

# create output files
run_name = datetime.today().strftime('%Y%m%d-%H%M%S') if run_name is None else run_name
output_dir_ver, version = get_latest_version(output_dir, run_name)
checkpoint_dir = os.path.join(output_dir_ver, 'checkpoints')
os.makedirs(output_dir_ver, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"RUN: {run_name}, VER: {version}")

# turn on tensorboard for convenience
print("Running TensorBoard...")
subprocess.run(["screen", "-dmS", "tb", "tensorboard", f"--logdir={output_dir}"]) 
print("Run 'screen -r tb' for TensorBoard.")
tb_logger = TensorBoardLogger(output_dir, name=run_name, version=version, default_hp_metric=False)

# get proper output dir and store hparams
tb_logger.log_hyperparams({
    "data": {
        "csv_file": csv_file,
        "train_hash_file": train_hash_file,
        "val_hash_file": val_hash_file,
        "output_dir": output_dir_ver,
        "precomputed_dir": precomputed_dir if loop_params["use_saved_reps_logs"] else "online training"},
    "loop_params": loop_params,
    "train_sampler_params": train_sampler_params,
    "checkpoint_params": checkpoint_params,
    "trainer_params": trainer_params})

# load modules
train_data_module = ProteinDataModule(csv_file, train_hash_file, train_sampler_params, collate_fn=lambda x: x)
val_data_module = ProteinDataModule(csv_file, val_hash_file, val_sampler_params, collate_fn=lambda x: x)
model_module = ProteinTrainModule(**loop_params, use_saved_reps_logs_dir=precomputed_dir, 
                                  output_dir=output_dir_ver, use_flash=USE_FA, distillation_loss=DistillationLoss())

# initialize trainer
trainer = pl.Trainer(
    **trainer_params,
    logger=tb_logger,
    devices=DEVICES,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    enable_progress_bar=True,  
    enable_model_summary=True,
    use_distributed_sampler=False,
    precision="bf16-mixed" if torch.cuda.is_bf16_supported() else 32,
    callbacks=[SetEpochCallback(), ModelCheckpoint(
        **checkpoint_params, dirpath=checkpoint_dir, 
        filename="checkpoint_{epoch}_{step}")])

# run training
train_data_module.setup()
val_data_module.setup()
trainer.fit(model_module, train_dataloaders=train_data_module.dataloader(), val_dataloaders=val_data_module.dataloader())
