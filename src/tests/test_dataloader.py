import os
import torch
import argparse
import yaml
import random
import numpy as np
from lightning.data_module import ProteinDataModule

def get_batches(data_module: ProteinDataModule):
    """Setup the data module and retrieve all batches from its dataloader."""
    data_module.setup()
    batches = []
    for batch in data_module.dataloader():
        batches.append(batch)
    return batches

def diff_batches(b1, b2, path="root"):
    """
    Recursively compare two batches and return a list of differences.
    Each difference is a string indicating the path and the mismatched values.
    If objects have a __dict__, their internal state is compared.
    """
    differences = []
    if type(b1) != type(b2):
        differences.append(f"Type mismatch at {path}: {type(b1)} vs {type(b2)}")
        return differences

    if isinstance(b1, torch.Tensor):
        if not torch.equal(b1, b2):
            differences.append(f"Tensor mismatch at {path}: shapes {b1.shape} vs {b2.shape}")
        return differences

    elif isinstance(b1, list):
        if len(b1) != len(b2):
            differences.append(f"List length mismatch at {path}: {len(b1)} vs {len(b2)}")
        for i, (a, b) in enumerate(zip(b1, b2)):
            differences.extend(diff_batches(a, b, f"{path}[{i}]"))
        if len(b1) > len(b2):
            for i in range(len(b2), len(b1)):
                differences.append(f"Extra element in first list at {path}[{i}]: {b1[i]}")
        elif len(b2) > len(b1):
            for i in range(len(b1), len(b2)):
                differences.append(f"Extra element in second list at {path}[{i}]: {b2[i]}")
        return differences

    elif isinstance(b1, dict):
        keys1 = set(b1.keys())
        keys2 = set(b2.keys())
        if keys1 != keys2:
            differences.append(f"Dict keys mismatch at {path}: {list(keys1)} vs {list(keys2)}")
        for k in keys1.intersection(keys2):
            differences.extend(diff_batches(b1[k], b2[k], f"{path}[{k}]"))
        for k in keys1 - keys2:
            differences.append(f"Key {k} present in first dict but missing in second at {path}")
        for k in keys2 - keys1:
            differences.append(f"Key {k} present in second dict but missing in first at {path}")
        return differences

    # If the objects have a __dict__, compare their internal state
    elif hasattr(b1, '__dict__') and hasattr(b2, '__dict__'):
        if b1.__dict__ != b2.__dict__:
            differences.append(f"Object state mismatch at {path}: {b1.__dict__} vs {b2.__dict__}")
        return differences

    else:
        if b1 != b2:
            differences.append(f"Value mismatch at {path}: {b1} vs {b2}")
        return differences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load the configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    csv_file = cfg["csv_file"]
    train_hash_file = cfg["train_hash_file"]
    train_sampler_params = cfg["train_sampler_params"]
    # Force single process for reproducibility checking
    train_sampler_params["num_replicas"] = 1
    train_sampler_params["rank"] = 0

    print("Collecting batches for first run...")
    data_module1 = ProteinDataModule(csv_file, train_hash_file, train_sampler_params, collate_fn=lambda x: x)
    batches1 = get_batches(data_module1)

    print("Collecting batches for second run...")
    data_module2 = ProteinDataModule(csv_file, train_hash_file, train_sampler_params, collate_fn=lambda x: x)
    batches2 = get_batches(data_module2)

    # Compare batches from both runs
    reproducible = True
    differences = []
    if len(batches1) != len(batches2):
        reproducible = False
        differences.append(f"Number of batches mismatch: {len(batches1)} vs {len(batches2)}")
    else:
        for idx, (b1, b2) in enumerate(zip(batches1, batches2)):
            diff = diff_batches(b1, b2, path=f"Batch {idx}")
            if diff:
                reproducible = False
                differences.append(f"Differences in batch {idx}:")
                differences.extend(diff)

    if reproducible:
        print("Batches are reproducible: both runs generated exactly the same batches.")
    else:
        print("Batches are NOT reproducible: differences found:")
        for line in differences:
            print(line)

if __name__ == "__main__":
    main()
