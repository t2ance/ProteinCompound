#!/usr/bin/env python3
"""
Randomly extract 1,000 samples from full dataset for quick testing.
"""
from datasets import load_from_disk
import random

# Paths
full_dataset_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_hf"
sample_dataset_path = "/home/peijia/projects/ProteinCompound/cache/sample_dataset"

print(f"Loading full dataset from {full_dataset_path}...")
full_dataset = load_from_disk(full_dataset_path)
num_samples = len(full_dataset)
print(f"Full dataset contains {num_samples:,} samples")

# Randomly select 1,000 indices
sample_size = 1000
print(f"\nRandomly selecting {sample_size:,} samples...")
random.seed(42)
all_indices = list(range(num_samples))
random.shuffle(all_indices)
sample_indices = sorted(all_indices[:sample_size])

# Select samples
print("Extracting samples...")
sample_dataset = full_dataset.select(sample_indices)

# Save to disk
print(f"\nSaving to {sample_dataset_path}...")
sample_dataset.save_to_disk(sample_dataset_path)

print(f"\nDone! Sample dataset with {len(sample_dataset):,} samples saved to:")
print(f"  {sample_dataset_path}")
print(f"\nUsage:")
print(f"  python main.py --hf-dataset {sample_dataset_path} --num-steps 100")
