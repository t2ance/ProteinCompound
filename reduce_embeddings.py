#!/usr/bin/env python3
"""
Reduce protein and compound embedding dimensions using PCA.

Protein: 1280 → k dimensions
Compound: 300 → k dimensions

Memory-efficient implementation for very large datasets.

nohup python reduce_embeddings.py > reduce_embeddings.log 2>&1 &
"""
import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
import random

# Paths
input_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_hf"
output_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_reduce_64_32"

# GPU settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PCA parameters
PROT_TARGET_DIM = 128  # Reduce from 1280 to k
COMP_TARGET_DIM = 32  # Reduce from 300 to k
FIT_SAMPLE_SIZE = 10000  # Number of samples to use for fitting PCA
FIT_BATCH_SIZE = 100  # Process 2000 samples at a time for fitting (with 73GB RAM available)

class PCAGPU:
    def __init__(self, n_components, device='cuda:0'):
        self.n_components = n_components
        self.device = torch.device(device)
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        X_torch = torch.from_numpy(X).float().to(self.device)
        n_samples, n_features = X_torch.shape
        self.n_samples_seen_ = n_samples

        self.mean_ = X_torch.mean(dim=0)
        X_centered = X_torch - self.mean_

        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[:self.n_components]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (n_samples - 1)
        total_var = ((X_centered ** 2).sum() / (n_samples - 1))
        self.explained_variance_ratio_ = (self.explained_variance_ / total_var).cpu().numpy()

        del X_torch, X_centered, U, S, Vt
        torch.cuda.empty_cache()
        return self

    def transform(self, X):
        X_torch = torch.from_numpy(X).float().to(self.device)
        X_centered = X_torch - self.mean_
        X_transformed = torch.mm(X_centered, self.components_.T)
        result = X_transformed.cpu().numpy()
        del X_torch, X_centered, X_transformed
        return result

print("=" * 80)
print("PCA Dimensionality Reduction for Protein-Compound Embeddings (GPU-Accelerated)")
print("=" * 80)
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
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

# Step 3: Load all fitting data efficiently using chunked processing
print("\nStep 3: Loading fitting data in chunks...")
prot_sequences_list = []
comp_sequences_list = []

LOAD_CHUNK_SIZE = 500
num_load_chunks = (len(fit_indices) + LOAD_CHUNK_SIZE - 1) // LOAD_CHUNK_SIZE

for chunk_idx in tqdm(range(num_load_chunks), desc="  Loading data"):
    start_idx = chunk_idx * LOAD_CHUNK_SIZE
    end_idx = min(start_idx + LOAD_CHUNK_SIZE, len(fit_indices))
    chunk_indices = fit_indices[start_idx:end_idx]

    chunk_samples = dataset.select(chunk_indices)
    for sample_prot, sample_comp in zip(chunk_samples['prot_emb'], chunk_samples['comp_emb']):
        prot_sequences_list.extend(sample_prot)
        comp_sequences_list.extend(sample_comp)

print("  Converting to numpy arrays...")
prot_sequences = np.array(prot_sequences_list, dtype=np.float32)
comp_sequences = np.array(comp_sequences_list, dtype=np.float32)
print(f"  Loaded {prot_sequences.shape[0]:,} protein sequences with {prot_sequences.shape[1]} dimensions")
print(f"  Loaded {comp_sequences.shape[0]:,} compound sequences with {comp_sequences.shape[1]} dimensions")

del prot_sequences_list, comp_sequences_list

# Step 4: Fit PCA models on GPU
print("\nStep 4: Fitting PCA models on GPU...")
prot_pca = PCAGPU(n_components=PROT_TARGET_DIM, device=device)
comp_pca = PCAGPU(n_components=COMP_TARGET_DIM, device=device)

print("  Fitting protein PCA...")
prot_pca.fit(prot_sequences)
del prot_sequences

print("  Fitting compound PCA...")
comp_pca.fit(comp_sequences)
del comp_sequences

torch.cuda.empty_cache()

# Print variance explained
prot_var_explained = np.sum(prot_pca.explained_variance_ratio_) * 100
comp_var_explained = np.sum(comp_pca.explained_variance_ratio_) * 100

print(f"\n  Protein PCA: {prot_var_explained:.2f}% variance explained with {PROT_TARGET_DIM} components")
print(f"  Compound PCA: {comp_var_explained:.2f}% variance explained with {COMP_TARGET_DIM} components")

# Step 5: Transform all samples using batched map (processes in chunks)
print(f"\nStep 5: Transforming all {num_samples:,} samples...")

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

# Apply transformation using batched processing (leverage 96 CPU cores)
reduced_dataset = dataset.map(
    transform_batch,
    batched=True,
    batch_size=500,  # Process 500 samples at a time
    desc="  Applying PCA",
    num_proc=16,  # Use 64 cores out of 96 available
)

# Step 6: Save reduced dataset
print(f"\nStep 6: Saving reduced dataset to {output_path}...")
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
