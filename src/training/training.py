import os
import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.pyl_utils import DistillationLoss, get_seq_rep, get_logits, batch_converter
from utils.pyl_utils import ProteinDataModule

import csv

class ProteinReprModule(pl.LightningModule):
    def __init__(self, student_model, teacher_model, distillation_loss, alphabet, repr_layer, batch_size, output_dir):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.distillation_loss = distillation_loss
        self.alphabet = alphabet
        self.repr_layer = repr_layer
        self.batch_size = batch_size
        self.output_dir = output_dir

    def forward(self, x):
        return self.student_model(x)



    def training_step(self, batch, batch_idx):
        sequences = [item['sequence'] for item in batch]
        names = [item['protein_id'] for item in batch]

        #  masking (always happens)
        masked_results = mask_batch(batch, batch_idx, self.current_epoch)
        masked_sequences = [masked_seq for masked_seq, _ in masked_results]
    
        #save masked sequences
        if self.save_masked_sequences:
            masked_sequences_dir = os.path.join(self.output_dir, 'masked_sequences')
            os.makedirs(masked_sequences_dir, exist_ok=True)
            
            with open(os.path.join(masked_sequences_dir, f"batch_{batch_idx}_masked_sequences.txt"), 'w') as f:
                for seq in masked_sequences:
                    f.write(seq + "\n")

        batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(names, masked_sequences)))
        batch_tokens = batch_tokens.to(self.device)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_res = self.teacher_model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)

        student_res = self.student_model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)

        #logits and representations
        teacher_logits = get_logits(teacher_res)
        teacher_reps = get_seq_rep(teacher_res, batch_lens, layer=self.repr_layer)
        student_logits = get_logits(student_res)
        student_reps = get_seq_rep(student_res, batch_lens, layer=self.repr_layer)

        # loss
        loss, rep_loss, log_loss = self.distillation_loss(teacher_reps, teacher_logits, student_reps, student_logits)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_rep_loss', rep_loss, prog_bar=True)
        self.log('train_log_loss', log_loss, prog_bar=True)

        # Define output directories for saving logits and representations
        teacher_logits_dir = os.path.join(self.output_dir, 'teacher_logits')
        teacher_reps_dir = os.path.join(self.output_dir, 'teacher_reps')
        student_logits_dir = os.path.join(self.output_dir, 'student_logits')
        student_reps_dir = os.path.join(self.output_dir, 'student_reps')

        # Ensure directories exist
        os.makedirs(teacher_logits_dir, exist_ok=True)
        os.makedirs(teacher_reps_dir, exist_ok=True)
        os.makedirs(student_logits_dir, exist_ok=True)
        os.makedirs(student_reps_dir, exist_ok=True)

        # Save teacher logits, representations, student logits, and representations
        torch.save(teacher_logits, os.path.join(teacher_logits_dir, f"batch_{batch_idx}_teacher_logits.pt"))
        torch.save(teacher_reps, os.path.join(teacher_reps_dir, f"batch_{batch_idx}_teacher_reps.pt"))
        torch.save(student_logits, os.path.join(student_logits_dir, f"batch_{batch_idx}_student_logits.pt"))
        torch.save(student_reps, os.path.join(student_reps_dir, f"batch_{batch_idx}_student_reps.pt"))

        return {"loss": loss, "train_rep_loss": rep_loss, "train_log_loss": log_loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=1e-4)

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        avg_rep_loss = torch.stack([x["train_rep_loss"] for x in outputs]).mean().item()
        avg_log_loss = torch.stack([x["train_log_loss"] for x in outputs]).mean().item()

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

# ---------------------- TRAINING ----------------------

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))

torch.set_float32_matmul_precision("medium")
torch.cuda.set_device(LOCAL_RANK)
DEVICES = torch.cuda.device_count()

model_type_student = "8M"
model_type_teacher = "650M"
csv_file = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/final_uniref100.csv"
hash_file = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/final_uniref100.hash"
output_dir_teacher = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/outputs/faesm_teacher_reps_teacher/"
output_dir_student = "/home/cpebiosustain_gmail_com/workspace/PLM_project_2.0/data/outputs/faesm_teacher_reps_student/"

sampler_params = {
    "num_replicas": DEVICES,
    "rank": RANK,
    "max_batch_tokens": 10000,
    "shuffle": False
}

data_module = ProteinDataModule(csv_file, hash_file, sampler_params, collate_fn=lambda x: x)

# Load teacher model
teacher_model = torch.load(f"{output_dir_teacher}/teacher_model.pth")
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False

# Load student model
student_model = ProteinReprModule(student_model=model_type_student, teacher_model=teacher_model,
                                  distillation_loss=DistillationLoss(), alphabet=None, repr_layer=12,
                                  output_dir=output_dir_student)

trainer = pl.Trainer(
    devices=DEVICES,
    accelerator="gpu",
    strategy="ddp",
    max_epochs=1,
    precision="bf16-mixed"
)

trainer.fit(student_model, datamodule=data_module)

