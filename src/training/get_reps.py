import pytorch_lightning as pl
from lightning.data_module import ProteinDataModule
from lightning.pyl_module import ProteinReprModule
import torch
import os



# GLOBAL ENV
RANK = int(os.environ.get("RANK", 0))
DEVICES = int(os.environ.get("WORLD_SIZE", 1))
torch.set_float32_matmul_precision("medium")



# MASTER PARAMS
model_type = "8M"
csv_file = '/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/uniprot_data_500k_sampled_250.csv'
hash_file = '/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/uniprot_data_500k_sampled_250.hash'
output_dir = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/outputs/faesm_teacher_reps/"

# SAMPLER PARAMS
sampler_params = {
    "num_replicas": DEVICES,
    "rank": RANK,
    "max_batch_size": None,
    "max_batch_tokens": 2200,
    "shuffle": False}



# init data module
data_module = ProteinDataModule(csv_file, hash_file, sampler_params, collate_fn=lambda x: x)
data_module.setup()

# init module for representations
model_module = ProteinReprModule(param_size=model_type, output_dir=output_dir)

# trainer
trainer = pl.Trainer(
    devices=DEVICES,
    accelerator="gpu",
    logger=False,
    precision="bf16-true",
    strategy="ddp",
    max_epochs=1,
    limit_test_batches=4) # ONLY ONE BATCH FOR TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# run the inference as test
trainer.test(model_module, dataloaders=data_module.dataloader(), verbose=False)
