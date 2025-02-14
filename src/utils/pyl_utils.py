import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.data_utils import HashedProteinDataset, DynamicTaxonIdSampler, ModelSelector, get_seq_rep


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, hash_file, sampler_params, collate_fn=lambda x: x):
        """
        :param csv_file: Path to CSV file.
        :param hash_file: Path to pre-hashed data (created via create_hashed_data).
        :param sampler_params: Dictionary with keys: num_replicas, rank, max_batch_size, max_batch_tokens, etc.
        :param collate_fn: Function for collating data.
        """
        super().__init__()
        self.csv_file = csv_file
        self.hash_file = hash_file
        self.sampler_params = sampler_params
        self.collate_fn = collate_fn

    def setup(self):
        self.dataset = HashedProteinDataset(self.csv_file, self.hash_file)

    def dataloader(self):
        sampler = DynamicTaxonIdSampler(
            num_replicas=self.sampler_params['num_replicas'],
            rank=self.sampler_params['rank'],
            seq_lengths=self.dataset.lengths,
            taxon_ids=self.dataset.taxon_ids,
            max_batch_size=self.sampler_params.get('max_batch_size', None),
            max_batch_tokens=self.sampler_params.get('max_batch_tokens', None),
            shuffle=self.sampler_params.get('shuffle', False)
        )
        sampler.dataset = self.dataset
        sampler.set_epoch(0)  # THIS MAKES EVERY EPOCH TO HAVE IDENTIDCAL BATCHES!!!!!!!!!!!!!!
        return DataLoader(self.dataset, batch_sampler=sampler, collate_fn=self.collate_fn, shuffle=False, num_workers=5)



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
        self.repr_layer = self.selector.repr
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
        rep = get_seq_rep(results, batch_lens, layers=self.repr_layer)
        return rep

    def test_step(self, batch, batch_idx):
        rep = self.forward(batch)
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"batch_{batch_idx+1}_reps.pt")
        torch.save(rep, save_path)
        return rep

    def configure_optimizers(self):
        return None
