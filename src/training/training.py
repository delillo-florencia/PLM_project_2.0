import os
import torch
import pytorch_lightning as pl
from utils.loss_functions import DistillationLoss
from tqdm import tqdm
from utils.data_utils import  get_seq_rep, get_logits, ModelSelector
from utils.token_mask import mask_batch, extract_masked_logits
from utils.pyl_utils import ProteinDataModule
import csv
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # or DEBUG for more output



class ProteinReprModule(pl.LightningModule):
    def __init__(self, student_model_param, teacher_model_param, distillation_loss, output_dir,save_masked_sequences):
        super().__init__()
        self.selector_student = ModelSelector(student_model_param)
        self.student_model = self.selector_student.model
        self.selector_teacher = ModelSelector(teacher_model_param)
        self.teacher_model = self.selector_teacher.model
        self.alphabet = self.selector_teacher.alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.distillation_loss = distillation_loss
        self.output_dir = output_dir
        self.teacher_model.eval()
        self.save_masked_sequences=save_masked_sequences
        self.outputs = []

        for param in self.teacher_model.parameters():
             param.requires_grad = False
    
    def forward(self, x):
        return self.student_model(x)
    
    def train_step(self, batch):
        rep = self.forward(batch)
        return rep


    def training_step(self, batch, batch_idx):

        #  masking 
        masked_results = mask_batch(batch, batch_idx, self.current_epoch)
        masked_batch, masked_pos = zip(*masked_results)
        print("masking done")
    
        #save masked sequences
        if self.save_masked_sequences:
            masked_sequences_dir = os.path.join(self.output_dir, 'masked_sequences')
            os.makedirs(masked_sequences_dir, exist_ok=True)
            
            with open(os.path.join(masked_sequences_dir, f"batch_{batch_idx}_masked_sequences.txt"), 'w') as f:
                for sample in masked_batch:
                    f.write(sample.masked_seq + "\n")

        print("masked seq save")
        
        masked_data = [(sample.seq_id, sample.masked_seq) for sample in masked_batch]
        _, _, batch_tokens = self.batch_converter(masked_data)
        masked_tokens = batch_tokens.to(self.device)  # Masked tokens for logits
        batch_lens = (masked_tokens != self.alphabet.padding_idx).sum(1)

        print("masked_tokens_ready")

        unmasked_data = [(sample.seq_id, sample.sequence) for sample in batch]  # Get unmasked sequences
        _, _, unmasked_tokens = self.batch_converter(unmasked_data)
        unmasked_tokens = unmasked_tokens.to(self.device)  # Unmasked tokens for representations
        print("unmasked_tokens_ready")

        print("-----------CHECK BELOW ------------------")
        print("masked_tokens dtype:", masked_tokens.dtype)
        print("masked_tokens shape:", masked_tokens.shape)
        #print("masked_tokens sample:", masked_tokens[:5])  

        #print("Batch tokens:", batch_tokens)
        print("Batch tokens shape:", batch_tokens.shape)
        print("Valid token indices range: 0 to", len(self.alphabet.all_toks) - 1)
        print("Unique token indices in masked_tokens:", masked_tokens.unique())
        print("LEN Unique token indices in masked_tokens:", len(masked_tokens.unique()))

        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()


        with torch.no_grad():
            teacher_res = self.teacher_model(unmasked_tokens)
        print("teacher_resok")
            
        student_res = self.student_model(unmasked_tokens)
        print("teacher and student res ok")
        teacher_reps = get_seq_rep(teacher_res, batch_lens)  
        student_reps = get_seq_rep(student_res, batch_lens)
        print("reps ok")
        print("-----------COMPARE BELOW ------------------")
        print("unmasked dtype:", unmasked_tokens.dtype)
        print("unmasked shape:", unmasked_tokens.shape)
        print("unmasked", len(unmasked_data), unmasked_data[0])
        print("masked_tokens dtype:", masked_tokens.dtype)
        print("masked_tokens shape:", masked_tokens.shape)
        print("masked", len(masked_data), masked_data[0])

        with torch.no_grad():
            teacher_res = self.teacher_model(masked_tokens)
        print("teacher_resok")
            
        student_res = self.student_model(masked_tokens)
        print("student_resok")
        student_logits = get_logits(student_res)  
        teacher_logits = get_logits(teacher_res)  
        print("getting_logits done")


        # loss
        student_logits = extract_masked_logits(student_logits, masked_pos)
        teacher_logits = extract_masked_logits(teacher_logits, masked_pos)
        print("logits masked positions ok")
        loss, rep_loss, log_loss = self.distillation_loss(teacher_reps, teacher_logits, student_reps, student_logits)
        print("LOSS OKKKK ok")

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_rep_loss', rep_loss, prog_bar=True)
        self.log('train_log_loss', log_loss, prog_bar=True)

        # Define output directories for saving logits and representations
        teacher_logits_dir = os.path.join(self.output_dir, 'teacher_logits')
        teacher_reps_dir = os.path.join(self.output_dir, 'teacher_reps')
        student_logits_dir = os.path.join(self.output_dir, 'student_logits')
        student_reps_dir = os.path.join(self.output_dir, 'student_reps')


        print("saving")
        # Save teacher logits, representations, student logits, and representations
        torch.save(teacher_logits, os.path.join(teacher_logits_dir, f"batch_{batch_idx}_teacher_logits.pt"))
        torch.save(teacher_reps, os.path.join(teacher_reps_dir, f"batch_{batch_idx}_teacher_reps.pt"))
        torch.save(student_logits, os.path.join(student_logits_dir, f"batch_{batch_idx}_student_logits.pt"))
        torch.save(student_reps, os.path.join(student_reps_dir, f"batch_{batch_idx}_student_reps.pt"))

        loss_dir = {"loss": loss, "train_rep_loss": rep_loss, "train_log_loss": log_loss}
        self.outputs.append(loss_dir)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=1e-4)

    def on_training_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.outputs]).mean().item()
        avg_rep_loss = torch.stack([x["train_rep_loss"] for x in self.outputs]).mean().item()
        avg_log_loss = torch.stack([x["train_log_loss"] for x in self.outputs]).mean().item()
        self.outputs.clear()

        self.log("avg_loss", avg_loss)
        self.log("avg_rep_loss", avg_rep_loss)
        self.log("avg_log_loss", avg_log_loss)

        log_file = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/outputs/training_log.csv"
        file_exists = os.path.exists(log_file)

        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "avg_loss", "avg_rep_loss", "avg_log_loss"])
            writer.writerow([self.current_epoch, avg_loss, avg_rep_loss, avg_log_loss])

    def on_train_epoch_end(self):
        if (self.current_epoch) % 5 == 0:
            checkpoint_path = f"{self.output_dir}/checkpoint_epoch_{self.current_epoch}.ckpt"
            torch.save({
                "epoch": self.current_epoch,
                "model_state_dict": self.student_model.state_dict(),
                "optimizer_state_dict": self.optimizers().optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

# DONT REMOVE PLS - ENSURE SET_EPOCH IS SET FOR EACH BATCH - REMOVE SET EPOCH FROM DATAMODULE
#class SetEpochCallback(pl.Callback):
#    def on_train_epoch_start(self, trainer, pl_module):
#        self.pl_module = pl_module
#        train_loader = trainer.train_dataloader
#        train_loader.batch_sampler.set_epoch(trainer.current_epoch)

# ---------------------- TRAINING ----------------------
RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
print("Init ok")
torch.set_float32_matmul_precision("high")
torch.cuda.set_device(LOCAL_RANK)
DEVICES = torch.cuda.device_count()

model_type_student = "8M"
model_type_teacher = "650M"
#csv_file = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/final_uniref100.csv"
#hash_file = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/final_uniref100.hash"
csv_file = '/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/uniprot_data_500k_sampled_250.csv'
hash_file = '/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/uniprot_data_500k_sampled_250.hash'
output_dir = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/outputs/"


# Define output directories for saving logits and representations
teacher_logits_dir = os.path.join(output_dir, 'teacher_logits')
teacher_reps_dir = os.path.join(output_dir, 'teacher_reps')
student_logits_dir = os.path.join(output_dir, 'student_logits')
student_reps_dir = os.path.join(output_dir, 'student_reps')

# Ensure directories exist
os.makedirs(teacher_logits_dir, exist_ok=True)
os.makedirs(teacher_reps_dir, exist_ok=True)
os.makedirs(student_logits_dir, exist_ok=True)
os.makedirs(student_reps_dir, exist_ok=True)
sampler_params = {
    "num_replicas": DEVICES,
    "rank": RANK,
    "max_batch_tokens": 10000,
    "shuffle": False, # all samples before bucketing
    "shuffle_batch_order": True, # batch order after bucketing
    "max_batch_num": 4, # max number of batches across all GPUs
}

data_module = ProteinDataModule(csv_file, hash_file, sampler_params, collate_fn=lambda x: x)
data_module.setup()

print("data module fine")
# Load  model
model = ProteinReprModule(student_model_param=model_type_student, teacher_model_param=model_type_teacher,
                                  distillation_loss=DistillationLoss(),output_dir=output_dir,save_masked_sequences=False)
print("model ok--")
trainer = pl.Trainer(
    devices=DEVICES,
    accelerator="gpu",
    strategy="ddp",
    max_epochs=1,
    enable_progress_bar=True,  
    log_every_n_steps=1,
    enable_model_summary=True,
    use_distributed_sampler=False,
    precision="bf16-mixed",
#    callbacks=[SetEpochCallback()]
)
print(torch.cuda.is_bf16_supported())  # Check BF16 support
print("Trainer ok")
trainer.fit(model, train_dataloaders=data_module.dataloader())
print("Done")
