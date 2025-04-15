
import os
import torch
import pytorch_lightning as pl
from models.model_selector import ModelSelector
from utils.model_utils import get_seq_rep

class ProteinReprModule(pl.LightningModule):
    def __init__(self, param_size="15B", output_dir="../../data/outputs/faesm_teacher_reps/"):
        """
        :param param_size: String specifying the model size (e.g. "15B").
        :param output_dir: Where to save the output representations.
        """
        super().__init__()
        self.selector = ModelSelector(param_size)
        self.model = self.selector.model  
        self.alphabet = self.selector.alphabet
        self.output_dir = output_dir
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device_type = torch.device("cuda")
        self.model.to(self.device_type)
        self.model.eval()

    def forward(self, batch):
        # Convert batch items to (seq_id, sequence)
        data = [(item.seq_id, item.sequence) for item in batch]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device_type)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.model(batch_tokens)
        rep = get_seq_rep(results, batch_lens)
        return rep

    def test_step(self, batch, batch_idx):
        rep = self.forward(batch)
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"batch_{batch_idx+1}_reps.pt")
        torch.save(rep, save_path)
        return rep

    def configure_optimizers(self):
        return None
