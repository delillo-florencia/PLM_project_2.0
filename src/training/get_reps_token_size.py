import os
import torch
import pytorch_lightning as pl
from utils.pyl_utils import ProteinDataModule, ProteinReprModule

# Let Slurm handle rank and world size
RANK = int(os.environ["SLURM_PROCID"])  # Global rank
WORLD_SIZE = int(os.environ["SLURM_NTASKS"])  # Total processes (should be 4 for 4 GPUs)
LOCAL_RANK = int(os.environ["SLURM_LOCALID"])  # Rank within the node
torch.set_float32_matmul_precision("medium")

torch.cuda.set_device(LOCAL_RANK)
print("-----------------------ok---------------------")
DEVICES = torch.cuda.device_count()  # Automatically detect available GPUs


# MASTER PARAMS
model_type = "650M"
csv_file = '/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/final_uniref100.csv'
hash_file = '/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/final_uniref100.hash'
output_dir = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/outputs/faesm_teacher_reps/"

# SAMPLER PARAMS
sampler_params = {
    "num_replicas": DEVICES,
    "rank": RANK,
    "max_batch_size": None,
    "max_batch_tokens": 2200,  # Default token size
    "shuffle": False}


# Initialize data module
data_module = ProteinDataModule(csv_file, hash_file, sampler_params, collate_fn=lambda x: x)
data_module.setup()

# Initialize module for representations
model_module = ProteinReprModule(param_size=model_type, output_dir=output_dir)


# Initialize the trainer
trainer = pl.Trainer(
    devices=DEVICES,
    accelerator="gpu",
    logger=False,
    precision="bf16-true",
    strategy="ddp",
    use_distributed_sampler=False,
    max_epochs=1,
    limit_test_batches=None)


# Define a range of max token sizes to test
#token_sizes = [1024, 1500, 2000, 2200, 2500, 3000,5000,7000,9000,10000,15000,20000]
#token_sizes=[40000,80000,100000,200000,350000,500000]
token_sizes=[10000]
# Loop through each token size
for tokens in token_sizes:
   # print(f"\nTesting with max_batch_tokens = {tokens}")
    
    # Update sampler params
    sampler_params["max_batch_tokens"] = tokens
    
    # Re-initialize the data module with the updated sampler params
    data_module = ProteinDataModule(csv_file, hash_file, sampler_params, collate_fn=lambda x: x)
    data_module.setup()
    
    # Reset GPU memory statistics
  #  torch.cuda.reset_peak_memory_stats()
    
    # Run the inference as test
    trainer.test(model_module, dataloaders=data_module.dataloader(), verbose=False)
    
    # Get memory usage
  #  memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
  #  max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
    
  #  print(f"Current Memory Allocated: {memory_allocated:.2f} GB")
  #  print(f"Max Memory Allocated: {max_memory_allocated:.2f} GB")

