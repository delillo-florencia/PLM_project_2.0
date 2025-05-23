import os
import torch
import re
import pytorch_lightning as pl
import logging
import torch.distributed as dist
from utils.model_utils import  get_seq_rep, get_logits, map_new_to_original
from models.model_selector import ModelSelector
from utils.token_mask import mask_batch, extract_masked_logits
from torch.nn.utils.rnn import pad_sequence

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)  # or DEBUG for more output


# trainer class
class ProteinTrainModule(pl.LightningModule):
    def __init__(self, student_model_param, teacher_model_param, distillation_loss, 
                 save_per_batch, use_saved_reps_logs, use_saved_reps_logs_dir, learning_rate,
                 output_dir,save_masked_sequences, save_reps_logs, use_flash, log_batch_shape=False):
        super().__init__()
        self.student_model = ModelSelector(student_model_param, use_flash).model
        self.teacher_model = ModelSelector(teacher_model_param, use_flash).model
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        self.teacher_model_param = teacher_model_param
        self.student_model.train()
        self.tokenizer = self.teacher_model.tokenizer
        self.distillation_loss = distillation_loss
        self.output_dir = output_dir
        self.save_masked_sequences=save_masked_sequences
        self.save_reps_logs = save_reps_logs
        self.save_per_batch = save_per_batch
        self.learning_rate = learning_rate
        self.use_saved_reps_logs_dir = use_saved_reps_logs_dir if use_saved_reps_logs else None
        self.log_batch_shape = log_batch_shape
        self.validation_step_outputs = []
        self.training_step_outputs = []
    
    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):

        # dataloader debugging for multi rank
        if self.log_batch_shape and dist.is_initialized():
            rank = dist.get_rank()
            batch_size = len(batch)
            lengths = [sample.length for sample in batch]
            taxon_ids = [sample.taxon_id for sample in batch]
            min_len, max_len = min(lengths), max(lengths)
            print(f"[Rank {rank}_train] Batch {batch_idx}: size={batch_size}, len=({min_len}-{max_len}), taxon_ids={set(taxon_ids)}")

        #  masking 
        masked_results = mask_batch(batch, batch_idx, self.current_epoch)
        masked_batch, masked_pos = zip(*masked_results)
    
        #save masked sequences
        if self.save_masked_sequences and self.current_epoch == 0:
            masks = [torch.tensor([1 if tok == "<mask>" else 0 for tok in re.findall(r"<mask>|.", s.masked_seq)],dtype=torch.bool) for s in masked_batch]
            tensor_mask = pad_sequence(masks, batch_first=True, padding_value=False)
            path = os.path.join(self.output_dir, "masking_tensors", f"rank-{dist.get_rank()}_batch-{batch_idx}-train_mask.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(tensor_mask, path)

        # prepare masked sequences
        masked_sequences = [sample.masked_seq for sample in masked_batch]
        masked_inputs = self.tokenizer(masked_sequences, return_tensors="pt", padding=True)
        masked_tokens = {k: v.to(self.device) for k, v in masked_inputs.items()}
        batch_lens = (masked_tokens["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        masked_pos = (masked_inputs["input_ids"] == self.tokenizer.mask_token_id)

        # prepare unmasked sequences
        unmasked_sequences = [sample.sequence for sample in batch]
        unmasked_inputs = self.tokenizer(unmasked_sequences, return_tensors="pt", padding=True)
        unmasked_tokens = {k: v.to(self.device) for k, v in unmasked_inputs.items()}

        ### TEACHER MODEL START ###

        # OFFLINE TRAINING
        if self.use_saved_reps_logs_dir:

            # compute mapping from (current_rank, batch_idx) → (orig_rank, orig_batch)
            orig_rank, orig_batch = map_new_to_original(
                new_rank        = dist.get_rank(),
                new_batch_idx   = batch_idx,
                orig_num_replicas = len(os.listdir(os.path.join(self.use_saved_reps_logs_dir, "teacher_reps"))),
                new_num_replicas  = self.trainer.world_size)

            # get teacher representations 
            teacher_reps = torch.load(os.path.join(
                self.use_saved_reps_logs_dir, "teacher_reps", f"rank-{orig_rank}",
                f"rank-{orig_rank}_batch-{orig_batch}-train_{self.teacher_model_param}-reps.pt"), 
                weights_only=True, map_location=torch.device(torch.cuda.current_device()))
            # get teacher logits
            teacher_logits = torch.load(os.path.join(
                self.use_saved_reps_logs_dir, "teacher_logits", f"rank-{orig_rank}",
                f"rank-{orig_rank}_batch-{orig_batch}-train_{self.teacher_model_param}-logi.pt"), 
                weights_only=True, map_location=torch.device(torch.cuda.current_device()))
            
        # ONLINE TRAINING
        else:
            # get teacher representations
            with torch.no_grad():
                teacher_res = self.teacher_model(**unmasked_tokens)
            teacher_reps = get_seq_rep(teacher_res, batch_lens)  
            # get teacher logits
            with torch.no_grad():
                teacher_res = self.teacher_model(**masked_tokens)
            teacher_logits = get_logits(teacher_res)
            teacher_logits = extract_masked_logits(teacher_logits, masked_pos)

        ### TEACHER MODEL END ###
                
        # get student representations
        student_res = self.student_model(**unmasked_tokens)
        student_reps = get_seq_rep(student_res, batch_lens)
        # get student logits
        student_res = self.student_model(**masked_tokens)
        student_logits = get_logits(student_res)  
        student_logits = extract_masked_logits(student_logits, masked_pos)

        # compute loss and store loss
        loss, rep_loss, log_loss = self.distillation_loss(teacher_reps, teacher_logits, student_reps, student_logits)
        self.training_step_outputs.append({
            "train_loss": loss.detach(),
            "train_reps_loss": rep_loss.detach(),
            "train_logi_loss": log_loss.detach()})

        #output directories for saving logits and representations per batch
        if self.current_epoch == 0 and self.save_reps_logs:
            teacher_logits_dir = os.path.join(self.output_dir, 'teacher_logits', f"rank-{dist.get_rank()}")
            teacher_reps_dir = os.path.join(self.output_dir, 'teacher_reps', f"rank-{dist.get_rank()}")
            os.makedirs(teacher_logits_dir, exist_ok=True)
            os.makedirs(teacher_reps_dir, exist_ok=True)
            torch.save(teacher_logits, os.path.join(teacher_logits_dir, f"rank-{dist.get_rank()}_batch-{batch_idx}-train_{self.teacher_model_param}-logi.pt"))
            torch.save(teacher_reps, os.path.join(teacher_reps_dir, f"rank-{dist.get_rank()}_batch-{batch_idx}-train_{self.teacher_model_param}-reps.pt"))

        # output for saving training logs per batch
        if self.save_per_batch:
            self.log("train_batch_size", len(batch), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            self.log("train_step_reps_loss", rep_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            self.log("train_step_logi_loss", log_loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        del teacher_res, student_res, teacher_reps, student_reps, teacher_logits, student_logits
        del masked_tokens, unmasked_tokens, masked_inputs, unmasked_inputs
        torch.cuda.empty_cache()
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)


    def on_train_epoch_end(self):

        # save metrics per epoch
        avg_loss = torch.stack([x["train_loss"] for x in self.training_step_outputs]).mean()
        avg_rep_loss = torch.stack([x["train_reps_loss"] for x in self.training_step_outputs]).mean()
        avg_log_loss = torch.stack([x["train_logi_loss"] for x in self.training_step_outputs]).mean()

        # log the aggregated losses
        self.log("train_epoch_loss", avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_epoch_reps_loss", avg_rep_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_epoch_logi_loss", avg_log_loss, on_epoch=True, prog_bar=False, sync_dist=True)

        # clear outputs list for the next epoch
        self.training_step_outputs.clear()
        torch.cuda.empty_cache()

    
    def validation_step(self, batch, batch_idx):

        # dataloader debugging for multi rank
        if self.log_batch_shape and dist.is_initialized():
            rank = dist.get_rank()
            batch_size = len(batch)
            lengths = [sample.length for sample in batch]
            taxon_ids = [sample.taxon_id for sample in batch]
            min_len, max_len = min(lengths), max(lengths)
            print(f"[Rank {rank}_val] Batch {batch_idx}: size={batch_size}, len=({min_len}-{max_len}), taxon_ids={set(taxon_ids)}")

        # Set teacher model to evaluation mode
        self.teacher_model.eval()

        # Prepare masked sequences
        masked_results = mask_batch(batch, batch_idx, self.current_epoch)
        masked_batch, _ = zip(*masked_results)
        masked_sequences = [sample.masked_seq for sample in masked_batch]
        masked_inputs = self.tokenizer(masked_sequences, return_tensors="pt", padding=True)
        masked_tokens = {k: v.to(self.device) for k, v in masked_inputs.items()}
        batch_lens = (masked_tokens["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        masked_pos = (masked_inputs["input_ids"] == self.tokenizer.mask_token_id)

        #save masked sequences
        if self.save_masked_sequences and self.current_epoch == 0:
            masks = [torch.tensor([1 if tok == "<mask>" else 0 for tok in re.findall(r"<mask>|.", s.masked_seq)],dtype=torch.bool) for s in masked_batch]
            tensor_mask = pad_sequence(masks, batch_first=True, padding_value=False)
            path = os.path.join(self.output_dir, "masking_tensors", f"rank-{dist.get_rank()}_batch-{batch_idx}-train_mask.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(tensor_mask, path)

        # Prepare unmasked sequences
        unmasked_sequences = [sample.sequence for sample in batch]
        unmasked_inputs = self.tokenizer(unmasked_sequences, return_tensors="pt", padding=True)
        unmasked_tokens = {k: v.to(self.device) for k, v in unmasked_inputs.items()}


        # OFFLINE VALIDATION
        if self.use_saved_reps_logs_dir:

            # compute mapping from (current_rank, batch_idx) → (orig_rank, orig_batch)
            orig_rank, orig_batch = map_new_to_original(
                new_rank        = dist.get_rank(),
                new_batch_idx   = batch_idx,
                orig_num_replicas = len(os.listdir(os.path.join(self.use_saved_reps_logs_dir, "teacher_reps"))),
                new_num_replicas  = self.trainer.world_size)

            # get teacher representations 
            teacher_reps = torch.load(os.path.join(
                self.use_saved_reps_logs_dir, "teacher_reps", f"rank-{orig_rank}",
                f"rank-{orig_rank}_batch-{orig_batch}-val_{self.teacher_model_param}-reps.pt"), 
                weights_only=True, map_location=torch.device(torch.cuda.current_device()))
            # get teacher logits
            teacher_logits = torch.load(os.path.join(
                self.use_saved_reps_logs_dir, "teacher_logits", f"rank-{orig_rank}",
                f"rank-{orig_rank}_batch-{orig_batch}-val_{self.teacher_model_param}-logi.pt"), 
                weights_only=True, map_location=torch.device(torch.cuda.current_device()))

        # ONLINE VALIDATION
        else:
            # get teacher representations
            with torch.no_grad():
                teacher_res = self.teacher_model(**unmasked_tokens)
            teacher_reps = get_seq_rep(teacher_res, batch_lens)
            # get teacher logits
            with torch.no_grad():
                teacher_res = self.teacher_model(**masked_tokens)
            teacher_logits = get_logits(teacher_res)
            teacher_logits = extract_masked_logits(teacher_logits, masked_pos)


        # Student: unmasked for representations, masked for logits
        student_res = self.student_model(**unmasked_tokens)
        student_reps = get_seq_rep(student_res, batch_lens)
        student_res = self.student_model(**masked_tokens)
        student_logits = get_logits(student_res)
        student_logits = extract_masked_logits(student_logits, masked_pos)

        # save validation batches
        if self.current_epoch == 0 and self.save_reps_logs:
            teacher_logits_dir = os.path.join(self.output_dir, 'teacher_logits', f"rank-{dist.get_rank()}")
            teacher_reps_dir = os.path.join(self.output_dir, 'teacher_reps', f"rank-{dist.get_rank()}")
            os.makedirs(teacher_logits_dir, exist_ok=True)
            os.makedirs(teacher_reps_dir, exist_ok=True)
            torch.save(teacher_logits, os.path.join(teacher_logits_dir, f"rank-{dist.get_rank()}_batch-{batch_idx}-val_{self.teacher_model_param}-logi.pt"))
            torch.save(teacher_reps, os.path.join(teacher_reps_dir, f"rank-{dist.get_rank()}_batch-{batch_idx}-val_{self.teacher_model_param}-reps.pt"))

        # compute and store loss
        loss, rep_loss, log_loss = self.distillation_loss(teacher_reps, teacher_logits, student_reps, student_logits)
        self.validation_step_outputs.append({
            "val_loss": loss.detach(),
            "val_reps_loss": rep_loss.detach(),
            "val_logi_loss": log_loss.detach()})
        
        # Log per step metrics if desired
        if self.save_per_batch:
            self.log("val_batch_size", len(batch), on_step=True, on_epoch=False, sync_dist=True)
            self.log("val_step_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=len(batch), sync_dist=True)
            self.log("val_step_reps_loss", rep_loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=len(batch), sync_dist=True)
            self.log("val_step_logi_loss", log_loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=len(batch), sync_dist=True)
            
        del teacher_res, student_res, teacher_reps, student_reps, teacher_logits, student_logits
        del masked_tokens, unmasked_tokens, masked_inputs, unmasked_inputs
        torch.cuda.empty_cache()
        return {"val_loss": loss, "val_reps_loss": rep_loss, "val_logi_loss": log_loss}
    

    def on_validation_epoch_end(self):

        # Aggregate outputs stored in validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_rep_loss = torch.stack([x["val_reps_loss"] for x in self.validation_step_outputs]).mean()
        avg_log_loss = torch.stack([x["val_logi_loss"] for x in self.validation_step_outputs]).mean()

        # Log aggregated validation metrics
        self.log("val_epoch_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_epoch_reps_loss", avg_rep_loss, prog_bar=False, sync_dist=True)
        self.log("val_epoch_logi_loss", avg_log_loss, prog_bar=False, sync_dist=True)

        # Clear the stored outputs for the next epoch
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()

