#!/usr/bin/env python3
import os
import sys
import argparse
import json
import yaml
from lightning.data_module import ProteinDataModule
from data.sampler import DynamicTaxonIdSampler

def precompute_split(cfg, split, world_size, root_out):
    """
    Precompute & shard batches for a given split ("train" or "val").
    """
    csv_file    = cfg[f"{split}_csv_file"]
    hash_file   = cfg[f"{split}_hash_file"]
    sampler_cfg = cfg["train_sampler_params"]

    if sampler_cfg["max_batch_num"] != -1 and split == "val":
        hash_params = sampler_cfg["hashing_params"]
        val_batch_num = int(sampler_cfg["max_batch_num"] * hash_params["val_size_ratio"] / hash_params["train_size_ratio"])
        sampler_cfg["max_batch_num"] = val_batch_num

    out_dir    = os.path.join(root_out, split)
    os.makedirs(out_dir, exist_ok=True)

    # Load dataset via your existing DataModule
    dm = ProteinDataModule(
        csv_file       = csv_file,
        hash_file      = hash_file,
        id_str         = split,
        sampler_params = sampler_cfg,
        collate_fn     = lambda x: x
    )
    dm.setup()
    ds = dm.dataset

    # Strip DDP‐only keys
    sampler_kwargs = {
        k: v for k, v in sampler_cfg.items()
        if k not in ("num_replicas", "rank")
    }

    # Precompute & shard
    DynamicTaxonIdSampler.precompute_batches(
        out_dir     = out_dir,
        seq_lengths = ds.lengths,
        taxon_ids   = ds.taxon_ids,
        id_str = f"precompte {split}",
        world_size  = world_size,
        **sampler_kwargs
    )

    meta = json.load(open(os.path.join(out_dir, "metadata.json")))
    print(f"→ {split.upper()} split: {meta['total_batches']} batches over {world_size} ranks")

def main():
    p = argparse.ArgumentParser(description="Precompute & shard sampler batches for TRAIN and VAL from your training YAML")
    p.add_argument("--config","-c", required=True, help="Path to your training YAML (the same one you pass to training_loop.py)")
    p.add_argument("--world_size", required=True, help="Number of total ranks for shredding")
    p.add_argument("--output_dir", required=True, help="Where to save shradded sampler")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if "precomputed_batches_dir" not in cfg:
        sys.exit("ERROR: please set `precomputed_batches_dir` in your YAML")

    # Do train and validation
    for split in ("train", "val"):
        precompute_split(cfg, split, int(args.world_size), args.output_dir)

    print("done")

if __name__=="__main__":
    main()
