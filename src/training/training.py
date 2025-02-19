import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from utils.pyl_utils import DistillationLoss, get_seq_rep, get_logits, batch_converter

class ProteinReprModule(pl.LightningModule):
    def __init__(self, student_model, teacher_model, distillation_loss, alphabet, repr_layer, batch_size, output_dir):
        super(ProteinReprModule, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.distillation_loss = distillation_loss
        self.alphabet = alphabet
        self.repr_layer = repr_layer
        self.batch_size = batch_size
        self.output_dir = output_dir

    def forward(self, x):
        # Forward pass through the student model
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        sequences = [item['sequence'] for item in batch]
        names = [item['protein_id'] for item in batch]

        # Data preparation
        data = list(zip(names, sequences))
        batch_seed = batch_idx * self.batch_size

        # Masking and data conversion
        with multiprocessing.Pool() as pool:
            masking = pool.starmap(mask_single, [(n, item, batch_seed) for n, item in enumerate(batch)]) 
        seqs, masked_pos = zip(*masking)

        data_mask = list(zip(names, seqs))

        # Convert data to batch tensors
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(self.device)

        # Convert masked data to batch tensors
        masked_batch_labels, masked_batch_strs, masked_batch_tokens = batch_converter(data_mask)
        masked_batch_lens = (masked_batch_tokens != self.alphabet.padding_idx).sum(1)
        masked_batch_tokens = masked_batch_tokens.to(self.device)

        # Forward pass (teacher and student models)
        teacher_logits = self.teacher_model(batch_tokens)
        teacher_reps = teacher_logits[0]  # Example assuming logits and reps are outputs

        # Forward pass for student
        student_res = self.student_model(batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        student_reps = get_seq_rep(student_res, batch_lens, layer=self.repr_layer)

        student_masked_res = self.student_model(masked_batch_tokens, repr_layers=[self.repr_layer], return_contacts=False)
        student_logits = get_logits(student_masked_res)

        masked_logi = []
        for i, positions in enumerate(masked_pos):
            positions = [i+1 for i in positions]  # Account for <str> token
            masked_logi.append(student_logits[i, positions, :])

        masked_student_logits = pad_sequence(masked_logi, batch_first=True, padding_value=0.0)

        # Compute loss
        loss, rep_loss, log_loss = self.distillation_loss(teacher_reps, teacher_logits, student_reps, masked_student_logits)

        # Log losses for backpropagation
        self.log('train_loss', loss)
        self.log('train_rep_loss', rep_loss)
        self.log('train_log_loss', log_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=1e-4)
        return optimizer

    def training_epoch_end(self, outputs):
        # Log metrics after each epoch
        avg_loss = torch.mean(torch.stack([x['loss'] for x in outputs])).item()
        avg_rep_loss = torch.mean(torch.stack([x['train_rep_loss'] for x in outputs])).item()
        avg_log_loss = torch.mean(torch.stack([x['train_log_loss'] for x in outputs])).item()

        self.log('avg_loss', avg_loss)
        self.log('avg_rep_loss', avg_rep_loss)
        self.log('avg_log_loss', avg_log_loss)

        print(f"Epoch {self.current_epoch + 1} - Loss: {avg_loss:.4f}, Representation Loss: {avg_rep_loss:.4f}, Log Loss: {avg_log_loss:.4f}")

    def on_epoch_end(self):
        # Save model checkpoints after each epoch (example logic for saving)
        if (self.current_epoch + 1) % 5 == 0:  # Save every 5 epochs (adjust frequency)
            checkpoint_path = f'{self.output_dir}/checkpoint_epoch_{self.current_epoch + 1}.ckpt'
            torch.save({
                'epoch': self.current_epoch + 1,
                'model_state_dict': self.student_model.state_dict(),
                'optimizer_state_dict': self.optimizers().state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
trainer = pl.Trainer(
    devices=DEVICES,
    accelerator="gpu",
    logger=False,  # You can enable logger if needed
    precision="bf16-true",
    strategy="ddp",
    max_epochs=10,  # Adjust as necessary
    limit_train_batches=1.0  # Or adjust to your needs (e.g., 0.5 for 50% of the data)
)

trainer.fit(model_module, dataloaders=data_module.dataloader())

