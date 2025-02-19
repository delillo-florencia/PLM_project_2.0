import csv
import pickle
import math
import random
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from faesm.esm import FAEsmForMaskedLM
from esm import Alphabet



class ModelSelector:

    MODEL_MAPPING = {
        "8M":   {"model_name": "facebook/esm2_t6_8M_UR50D"},
        "35M":  {"model_name": "facebook/esm2_t12_35M_UR50D"},
        "150M": {"model_name": "facebook/esm2_t30_150M_UR50D"},
        "650M": {"model_name": "facebook/esm2_t33_650M_UR50D"},
        "3B":   {"model_name": "facebook/esm2_t36_3B_UR50D"},
        "15B":  {"model_name": "facebook/esm2_t48_15B_UR50D"}}
    
    def __init__(self, param_size: str):
        try:
            model_info = self.MODEL_MAPPING[param_size]
        except KeyError:
            raise ValueError(param_size)
        self.model = FAEsmForMaskedLM.from_pretrained(model_info["model_name"])
        self.alphabet = Alphabet.from_architecture("ESM-1")



class Sequence:
    
    def __init__(self, sequence, length, taxon_id, seq_id):
        self.sequence = str(sequence)
        self.length = int(length)
        self.taxon_id = int(taxon_id)
        self.seq_id = str(seq_id)



class HashedProteinDataset(Dataset):
    """
    Firstly, pre-hash your master dataset:
    > HashedProteinDataset.create_hashed_data("massive_file.csv", "massive_file.hash")

    Later, use hashed file to generate the dataset with hashed reference:
    dataset = HashedProteinDataset("massive_file.csv", "massive_file.hash")
    """
    def __init__(self, csv_file, hashed_data_path):
        with open(hashed_data_path, 'rb') as f:
            data = pickle.load(f)
        self.header = data['header']
        self.line_offsets = data['line_offsets']
        self.lengths = data['lengths']
        self.taxon_ids = data['taxon_ids']
        self.csv_file = csv_file
        self._file_handle = None

    def __len__(self):
        return len(self.line_offsets)

    def _get_file_handle(self):
        if self._file_handle is None:
            self._file_handle = open(self.csv_file, 'r', newline='', encoding='utf-8')
        return self._file_handle

    def __getitem__(self, index):
        f = self._get_file_handle()
        f.seek(self.line_offsets[index])
        line = f.readline().strip()
        row = line.split(',')
        return Sequence(row[3], row[2], row[1], row[0])

    @staticmethod
    def create_hashed_data(csv_file, hashed_data_path):
        offsets = []
        lengths = []
        taxon_ids = []
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            header = next(csv.reader([f.readline()]))
            pos = f.tell()
            line = f.readline()
            while line:
                offsets.append(pos)
                row = line.strip().split(',')
                d = dict(zip(header, row))
                lengths.append(int(d['sequence_length']))
                taxon_ids.append(int(d['taxon_id']))
                pos = f.tell()
                line = f.readline()
        data = {'header': header, 'line_offsets': offsets, 'lengths': lengths, 'taxon_ids': taxon_ids}
        with open(hashed_data_path, 'wb') as f:
            pickle.dump(data, f)



class DynamicTaxonIdSampler(Sampler):
    def __init__(self, num_replicas, rank, seq_lengths, taxon_ids, num_buckets=128, min_len=0, max_len=1024,
                 max_batch_tokens=None, max_batch_size=None, shuffle=False, shuffle_batch_order=True, seed=42, drop_last=False):
        """
        A dynamic batch sampler supports DDP for robust training
        :param num_replicas: int
            the world size (i.e. the num of gpus), set it to 1 if you are using single gpu
        :param rank: int
            the rank of the gpu (see PyTorch DDP docs for details), set it to 0 if you are using single gpu
        :param seq_lengths: list
        :param taxon_ids: list
        :param num_buckets: int
            the smaller the num_buckets, the richer the permutation in one batch.
        :param min_len: int
            skip the sample whose length < min_len
        :param max_len: int
            skip the sample whose length > max_len
        :param max_batch_tokens: int or None
            max_batch_tokens and max_batch_size determine the usage of gpu memory and the 'real batch size' together
        :param max_batch_size: int or None
            max_batch_size and max_batch_tokens determine the usage of gpu memory and the 'real batch size' together
        :param shuffle: bool
        :param seed: int
        :param drop_last: bool
        """
        super().__init__(None)
        self.num_replicas = num_replicas
        self.rank = rank # FIX RANK TO AUTO-MODE
        self.seq_lengths = seq_lengths
        self.taxon_ids = taxon_ids
        self.num_buckets = num_buckets
        self.min_len = min_len
        self.max_len = max_len
        self.max_batch_tokens = max_batch_tokens or float('inf')
        self.max_batch_size = (max_batch_size + 1) if max_batch_size is not None else float('inf')
        self.shuffle = shuffle
        self.shuffle_batch_order = shuffle_batch_order
        self.seed = seed
        self.drop_last = drop_last
        self.__epoch = 0
        self.__per_gpu_batch_num = 0
        self.__batches = []
        random.seed(seed)

    def __len__(self):
        return self.__per_gpu_batch_num

    def __iter__(self):
        for batch in self.__batches[self.rank::self.num_replicas]:
            yield batch

    def set_epoch(self, epoch):
        self.__epoch = epoch
        self.__batches = self._prepare_batches()

    def _is_full(self, tokens, batch):
        return len(batch) >= self.max_batch_size or tokens > self.max_batch_tokens

    def _prepare_batches(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.__epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            if len(self.__batches) > 0:
                return self.__batches
            indices = list(range(len(self.dataset)))

        batches = []
        buckets = {}  # key: (taxon_id, length_bucket)

        for idx in indices:
            length = self.seq_lengths[idx]
            taxon_id = self.taxon_ids[idx]
            if not (self.min_len <= length <= self.max_len):
                continue

            # Compute a length bucket and use taxon_id to form the bucket key.
            bucket_key = (taxon_id, math.floor((length - self.min_len) / (self.max_len - self.min_len + 1) * self.num_buckets))
            if bucket_key not in buckets:
                buckets[bucket_key] = {'indices': [], 'max_len': 0}
            bucket = buckets[bucket_key]

             # Append the sample first.
            bucket['indices'].append(idx)
            bucket['max_len'] = max(bucket['max_len'], length)
            tokens = len(bucket['indices']) * bucket['max_len']

            # Check if the bucket has become full.
            if self._is_full(tokens, bucket['indices']):
                if len(bucket['indices']) > 1:
                    # Pop the last sample and flush the rest.
                    last = bucket['indices'].pop()
                    batches.append(bucket['indices'])
                    # Start a new bucket with the popped sample.
                    buckets[bucket_key] = {'indices': [last], 'max_len': length}
                else:
                    batches.append(bucket['indices'])
                    buckets[bucket_key] = {'indices': [], 'max_len': 0}

        # process leftover samples in all buckets.
        for bucket in buckets.values():
            if bucket['indices']:
                batches.append(bucket['indices'])

        # make sure the number of butches works for DDP
        random.seed(self.seed + self.__epoch)
        per_gpu = math.ceil(len(batches) / self.num_replicas)
        total = per_gpu * self.num_replicas
        dummy = total - len(batches)
        if dummy <= len(batches):
            batches += random.sample(batches, dummy)
        else:
            batches += [random.choice(batches) for _ in range(dummy)]

        # shuffle batch order
        if self.shuffle_batch_order:
            rng = np.random.default_rng(self.seed + self.__epoch)
            permuted_order = rng.permutation(len(batches))
            batches = [batches[i] for i in permuted_order]

        self.__per_gpu_batch_num = per_gpu
        return batches
    


def get_seq_rep(results, batch_lens):
    """
    Get sequence representations from esm_compute
    """
    token_representations = results["last_hidden_state"]

    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    return sequence_representations



def get_logits(results):
    """
    Extracts logits from esm_compute
    """
    logits = results["logits"]  

    return logits

