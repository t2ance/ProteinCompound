#!/usr/bin/env python3
"""
Reduce protein and compound embedding dimensions using PCA.
Protein: 1280 → 128 dimensions, Compound: 300 → 32 dimensions

nohup python reduce_embeddings.py > reduce_embeddings.log 2>&1 &
"""
import numpy as np
import torch
from torch_pca import PCA
from datasets import load_from_disk
from tqdm import tqdm
import random
import time

# Configuration
input_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_hf"
output_path = "/home/peijia/projects/ProteinCompound/cache/embeddings_reduce_128_32"
PROT_DIM, COMP_DIM = 128, 32
FIT_SAMPLES = 2000
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
dataset = load_from_disk(input_path)
print(f"Dataset: {len(dataset):,} samples")

# Sample and load fitting data
random.seed(42)
fit_idx = sorted(random.sample(range(len(dataset)), FIT_SAMPLES))
prot_data, comp_data = [], []
for idx in tqdm(fit_idx, desc="Loading fit data"):
    sample = dataset[idx]
    prot_data.extend(sample['prot_emb'])
    comp_data.extend(sample['comp_emb'])

prot_tensor = torch.tensor(prot_data, dtype=torch.float32, device=device)
comp_tensor = torch.tensor(comp_data, dtype=torch.float32, device=device)
print(f"Fit data: {prot_tensor.shape[0]:,} protein seqs, {comp_tensor.shape[0]:,} compound seqs")

# Fit PCA on GPU
print("Fitting PCA models on GPU...")
prot_pca = PCA(n_components=PROT_DIM).fit(prot_tensor)
comp_pca = PCA(n_components=COMP_DIM).fit(comp_tensor)
print(f"Protein PCA: {prot_pca.explained_variance_ratio_.sum()*100:.2f}% variance")
print(f"Compound PCA: {comp_pca.explained_variance_ratio_.sum()*100:.2f}% variance")

# Move PCA models to CPU for multiprocessing compatibility
prot_pca_cpu = PCA(n_components=PROT_DIM)
prot_pca_cpu.mean_ = prot_pca.mean_.cpu()
prot_pca_cpu.components_ = prot_pca.components_.cpu()
prot_pca_cpu.explained_variance_ = prot_pca.explained_variance_.cpu()
prot_pca_cpu.explained_variance_ratio_ = prot_pca.explained_variance_ratio_.cpu()

comp_pca_cpu = PCA(n_components=COMP_DIM)
comp_pca_cpu.mean_ = comp_pca.mean_.cpu()
comp_pca_cpu.components_ = comp_pca.components_.cpu()
comp_pca_cpu.explained_variance_ = comp_pca.explained_variance_.cpu()
comp_pca_cpu.explained_variance_ratio_ = comp_pca.explained_variance_ratio_.cpu()

del prot_tensor, comp_tensor, prot_pca, comp_pca

# Transform on CPU with multiprocessing
def transform_batch(batch):
    t0 = time.time()
    # Flatten and create CPU tensor
    all_prot = torch.FloatTensor([emb for sample in batch['prot_emb'] for emb in sample])
    all_prot_reduced = prot_pca_cpu.transform(all_prot)

    # Unflatten back to original structure
    prot_embs = []
    idx = 0
    for sample in batch['prot_emb']:
        n = len(sample)
        prot_embs.append(all_prot_reduced[idx:idx+n].numpy().tolist())
        idx += n

    # Same for compounds
    all_comp = torch.FloatTensor([emb for sample in batch['comp_emb'] for emb in sample])
    all_comp_reduced = comp_pca_cpu.transform(all_comp)

    comp_embs = []
    idx = 0
    for sample in batch['comp_emb']:
        n = len(sample)
        comp_embs.append(all_comp_reduced[idx:idx+n].numpy().tolist())
        idx += n

    return {
        'prot_emb': prot_embs,
        'comp_emb': comp_embs,
        'prot_mask': batch['prot_mask'],
        'comp_mask': batch['comp_mask'],
        'label': batch['label']
    }

print("Transforming with parallel CPU processing...")
reduced_dataset = dataset.map(transform_batch, batched=True, batch_size=1000, num_proc=2, desc="Transforming")
reduced_dataset.save_to_disk(output_path)
print(f"Done! Saved to {output_path}")
