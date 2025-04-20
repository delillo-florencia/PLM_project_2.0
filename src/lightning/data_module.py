import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import HashedProteinDataset
from data.sampler import DynamicTaxonIdSampler
import os

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, hash_file, id_str, sampler_params, collate_fn=lambda x: x):
        """
        :param csv_file: Path to CSV file.
        :param hash_file: Path to pre-hashed data (created via create_hashed_data).
        :param sampler_params: Dictionary with keys: num_replicas, rank, max_batch_size, max_batch_tokens, etc.
        :param collate_fn: Function for collating data.
        """
        super().__init__()
        self.csv_file = csv_file
        self.hash_file = hash_file
        self.id = id_str
        self.sampler_params = sampler_params
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        self.stage = stage
        self.dataset = HashedProteinDataset(self.csv_file, self.hash_file)

    def dataloader(self):
        sampler = DynamicTaxonIdSampler(
            **self.sampler_params,
            seq_lengths=self.dataset.lengths,
            taxon_ids=self.dataset.taxon_ids,
            id_str=self.id
        )
        sampler.dataset = self.dataset
        sampler.set_epoch(0)  # this is here only to initialize, epochs ar eupdated with a callback, see training :)
        return DataLoader(self.dataset, batch_sampler=sampler, collate_fn=self.collate_fn, shuffle=False, num_workers=int(os.environ["OMP_NUM_THREADS"]))


