from torch.utils.data import  Sampler
import random
import torch
import math
import numpy as np
import pickle
import sys
import os
import json
from tqdm import tqdm as tqdm

# disable dynamic bars if not a terminal (or force-leave short final line)
#def tqdm(*args, **kwargs):
#    disable = not sys.stdout.isatty()
#    return _tqdm(*args, disable=disable, leave=False, **kwargs)

class DynamicTaxonIdSampler(Sampler):
    def __init__(self, num_replicas, rank, seq_lengths, taxon_ids, id_str, precomputed_batches_dir=None, num_buckets=64, min_len=0, max_len=1024, max_batch_num=None,
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
        self.rank = rank
        self.seq_lengths = seq_lengths
        self.taxon_ids = taxon_ids
        self.num_buckets = num_buckets
        self.min_len = min_len
        self.max_len = max_len
        self.max_batch_tokens = max_batch_tokens or float('inf')
        self.max_batch_size = (max_batch_size + 1) if max_batch_size is not None else float('inf')
        self.max_batch_num = max_batch_num if max_batch_num != -1 else float('inf')
        self.shuffle = shuffle
        self.shuffle_batch_order = shuffle_batch_order
        self.seed = seed
        self.drop_last = drop_last
        self.fixed_batches = None
        self.id = id_str
        self.precomputed_batches = precomputed_batches_dir
        self.__epoch = 0
        self.__per_gpu_batch_num = 0
        self.__batches = []
        random.seed(seed)

    def __len__(self):
        return self.__per_gpu_batch_num

    def __iter__(self):
        if self.precomputed_batches is not None:
            for batch in self.__batches:
                yield batch
        else:
            for batch in self.__batches[self.rank::self.num_replicas]:
                yield batch

    def set_epoch(self, epoch):
        self.__epoch = epoch

        # If max_batch_num is set and fixed_batches is already computed, reuse them
        if self.fixed_batches is not None:
            rng = np.random.default_rng(self.seed + self.__epoch)
            permuted_order = rng.permutation(len(self.fixed_batches))
            self.__batches = [self.fixed_batches[i] for i in permuted_order]
            self.__per_gpu_batch_num = math.ceil(len(self.__batches) / self.num_replicas)
            return

        # load precomputed batches if desired
        if self.precomputed_batches is not None:
            path = os.path.join(self.precomputed_batches, f"batches_shard_{self.rank}.pkl")
            print(f"[rank {self.rank}] Loaded precomputed batches.")
            with open(path, "rb") as f:
                batches = pickle.load(f)
            self.fixed_batches = batches.copy()
            self.__batches = batches
            self.__per_gpu_batch_num = len(batches)
            return

        # build batches on the fly
        batches = self._prepare_batches()
        self.fixed_batches = batches.copy()
        self.__batches = batches
        self.__per_gpu_batch_num = math.ceil(len(batches) / self.num_replicas)


    def _is_full(self, tokens, batch):
        return len(batch) >= self.max_batch_size or tokens > self.max_batch_tokens

    def _prepare_batches(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.__epoch)
            indices = torch.randperm(len(self.seq_lengths), generator=g).tolist()
        else:
            if len(self.__batches) > 0:
                return self.__batches
            indices = list(range(len(self.seq_lengths)))

        batches = []
        buckets = {}  # key: (taxon_id, length_bucket)
        for idx in tqdm(indices, desc=f"[rank {self.rank}] Processing {self.id} batches", position=self.rank, unit="samples"):

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
                if not self.drop_last or len(bucket['indices']) >= self.max_batch_size:
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
            rng.shuffle(batches)

        # truncate the total number of batches to max_batch_num (total over all GPUs)
        # THIS LOGIC IS CORRECT - SEE _SET_EPOCH OK MAN
        if self.max_batch_num != float('inf'):
            total = (self.max_batch_num // self.num_replicas) * self.num_replicas
            batches = batches[:total]
            per_gpu = total // self.num_replicas
        else:
            per_gpu = math.ceil(len(batches) / self.num_replicas)

        if self.max_batch_size != float('inf'):
            padded_batches = []
            for batch in batches:
                batch_len = len(batch)
                if batch_len < self.max_batch_size:
                    pad_count = self.max_batch_size - batch_len
                    # Duplicate real samples from the same batch
                    padded = batch + random.choices(batch, k=pad_count)
                    padded_batches.append(padded)
                elif batch_len > self.max_batch_size:
                    # This should rarely happen, but just clip to max_batch_size
                    padded_batches.append(batch[:self.max_batch_size])
                else:
                    padded_batches.append(batch)
            batches = padded_batches
    
        self.__per_gpu_batch_num = per_gpu
        return batches

    @staticmethod
    def precompute_batches(out_dir, seq_lengths, taxon_ids, world_size, **sampler_kwargs):

        os.makedirs(out_dir, exist_ok=True)

        # 1) Build global batches
        sampler = DynamicTaxonIdSampler(
            num_replicas=1, rank=0,
            seq_lengths=seq_lengths,
            taxon_ids=taxon_ids,
            **sampler_kwargs)
        global_batches = sampler._prepare_batches()

        # 2) Shard
        per_shard = math.ceil(len(global_batches) / world_size)
        for rank in range(world_size):
            start = rank * per_shard
            end   = min(start + per_shard, len(global_batches))
            shard = global_batches[start:end]
            with open(f"{out_dir}/batches_shard_{rank}.pkl", "wb") as f:
                pickle.dump(shard, f)

        # 3) Write metadata
        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump({
                "total_batches":   len(global_batches),
                "world_size":      world_size,
                "sampler_params":  sampler_kwargs,
            }, f, indent=2)