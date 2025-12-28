#!/usr/bin/env python3
"""
Reduce protein and compound embedding dimensions using PCA.

Protein: 1280 → 32 dimensions
Compound: 300 → 32 dimensions

Memory-efficient implementation for very large datasets.
"""
import numpy as np
from sklearn.decomposition import IncrementalPCA
from datasets import load_from_disk
from tqdm import tqdm
import random

# Paths
input_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_hf"
output_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_reduce"

# PCA parameters
PROT_TARGET_DIM = 32  # Reduce from 1280 to 32
COMP_TARGET_DIM = 32  # Reduce from 300 to 32
FIT_SAMPLE_SIZE = 10000  # Number of samples to use for fitting PCA
FIT_BATCH_SIZE = 50  # Process 50 samples at a time for fitting (memory-efficient)

print("=" * 80)
print("PCA Dimensionality Reduction for Protein-Compound Embeddings")
print("=" * 80)
print(f"\nInput:  {input_path}")
print(f"Output: {output_path}")
print(f"\nTarget dimensions: Protein {PROT_TARGET_DIM}, Compound {COMP_TARGET_DIM}")
print(f"Fitting PCA on {FIT_SAMPLE_SIZE:,} random samples (batches of {FIT_BATCH_SIZE})\n")

# Step 1: Load dataset (memory-mapped, doesn't load into RAM)
print("Step 1: Loading dataset (memory-mapped)...")
dataset = load_from_disk(input_path)
num_samples = len(dataset)
print(f"  Dataset contains {num_samples:,} samples")

# Step 2: Sample indices for PCA fitting
print(f"\nStep 2: Sampling {FIT_SAMPLE_SIZE:,} random indices for PCA fitting...")
random.seed(42)
all_indices = list(range(num_samples))
random.shuffle(all_indices)
fit_indices = sorted(all_indices[:FIT_SAMPLE_SIZE])
print(f"  Selected indices: {fit_indices[0]} to {fit_indices[-1]}")

# Step 3: Fit PCA models using IncrementalPCA (batch-by-batch)
print("\nStep 3: Fitting PCA models in small batches (memory-efficient)...")

prot_pca = IncrementalPCA(n_components=PROT_TARGET_DIM)
comp_pca = IncrementalPCA(n_components=COMP_TARGET_DIM)

num_batches = (len(fit_indices) + FIT_BATCH_SIZE - 1) // FIT_BATCH_SIZE
print(f"  Processing {num_batches} batches of {FIT_BATCH_SIZE} samples each")

for batch_idx in tqdm(range(num_batches), desc="  Fitting PCA"):
    start_idx = batch_idx * FIT_BATCH_SIZE
    end_idx = min(start_idx + FIT_BATCH_SIZE, len(fit_indices))
    batch_indices = fit_indices[start_idx:end_idx]

    # Process samples one by one to avoid loading too much at once
    prot_sequences = []
    comp_sequences = []

    for idx in batch_indices:
        sample = dataset[idx]
        prot_sequences.extend(sample['prot_emb'])
        comp_sequences.extend(sample['comp_emb'])

    # Partial fit PCA models (only loads current batch into memory)
    if prot_sequences:
        prot_pca.partial_fit(np.array(prot_sequences, dtype=np.float32))
        del prot_sequences  # Free memory immediately

    if comp_sequences:
        comp_pca.partial_fit(np.array(comp_sequences, dtype=np.float32))
        del comp_sequences  # Free memory immediately

# Print variance explained
prot_var_explained = np.sum(prot_pca.explained_variance_ratio_) * 100
comp_var_explained = np.sum(comp_pca.explained_variance_ratio_) * 100

print(f"\n  Protein PCA: {prot_var_explained:.2f}% variance explained with {PROT_TARGET_DIM} components")
print(f"  Compound PCA: {comp_var_explained:.2f}% variance explained with {COMP_TARGET_DIM} components")

# Step 4: Transform all samples using batched map (processes in chunks)
print(f"\nStep 4: Transforming all {num_samples:,} samples...")

def transform_batch(batch):
    """Apply PCA transformation to a batch of samples (memory-efficient)."""
    prot_embs_reduced = []
    comp_embs_reduced = []

    for i in range(len(batch['prot_emb'])):
        # Transform protein embeddings for this sample
        prot_emb_np = np.array(batch['prot_emb'][i], dtype=np.float32)
        prot_reduced = prot_pca.transform(prot_emb_np)
        prot_embs_reduced.append(prot_reduced.tolist())

        # Transform compound embeddings for this sample
        comp_emb_np = np.array(batch['comp_emb'][i], dtype=np.float32)
        comp_reduced = comp_pca.transform(comp_emb_np)
        comp_embs_reduced.append(comp_reduced.tolist())

    return {
        'prot_emb': prot_embs_reduced,
        'prot_mask': batch['prot_mask'],
        'comp_emb': comp_embs_reduced,
        'comp_mask': batch['comp_mask'],
        'label': batch['label'],
    }

# Apply transformation using batched processing (memory-efficient)
reduced_dataset = dataset.map(
    transform_batch,
    batched=True,
    batch_size=100,  # Process 100 samples at a time
    desc="  Applying PCA",
    num_proc=1,  # Single process to avoid memory issues
)

# Step 5: Save reduced dataset
print(f"\nStep 5: Saving reduced dataset to {output_path}...")
reduced_dataset.save_to_disk(output_path)

print("\n" + "=" * 80)
print("DONE! Summary:")
print("=" * 80)
print(f"  Input samples:  {num_samples:,}")
print(f"  Output samples: {len(reduced_dataset):,}")
print(f"  Protein embedding: 1280 → {PROT_TARGET_DIM} dims ({prot_var_explained:.2f}% variance)")
print(f"  Compound embedding: 300 → {COMP_TARGET_DIM} dims ({comp_var_explained:.2f}% variance)")
print(f"\n  Saved to: {output_path}")
print("\nVerification:")
print(f"  Check size: du -sh {output_path}")
print(f"  Load test: python -c \"from datasets import load_from_disk; d=load_from_disk('{output_path}'); print(len(d[0]['prot_emb'][0]), len(d[0]['comp_emb'][0]))\"")
print()
