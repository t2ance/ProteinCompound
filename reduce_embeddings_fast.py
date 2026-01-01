#!/usr/bin/env python3
"""
Optimized PCA reduction - processes data in streaming fashion to avoid memory explosion
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
BATCH_SIZE = 100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f"Device: {device}")
dataset = load_from_disk(input_path)
print(f"Dataset: {len(dataset):,} samples")

# Fit PCA
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

print("Fitting PCA on GPU...")
prot_pca = PCA(n_components=PROT_DIM).fit(prot_tensor)
comp_pca = PCA(n_components=COMP_DIM).fit(comp_tensor)
print(f"Protein PCA: {prot_pca.explained_variance_ratio_.sum()*100:.2f}% variance")
print(f"Compound PCA: {comp_pca.explained_variance_ratio_.sum()*100:.2f}% variance")
del prot_tensor, comp_tensor, prot_data, comp_data

# Manual batch processing with GPU
print("Transforming batches...")
all_results = []
n_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE

with tqdm(total=len(dataset), desc="Transforming") as pbar:
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(dataset))

        # Load batch (this is slow due to Arrow deserialization)
        batch_data = dataset[start_idx:end_idx]

        # Process on GPU (fast)
        with torch.no_grad():
            # Protein
            all_prot = torch.FloatTensor([emb for sample in batch_data['prot_emb'] for emb in sample]).to(device)
            all_prot_reduced = prot_pca.transform(all_prot).cpu()

            # Compound
            all_comp = torch.FloatTensor([emb for sample in batch_data['comp_emb'] for emb in sample]).to(device)
            all_comp_reduced = comp_pca.transform(all_comp).cpu()

        # Unflatten
        prot_embs, comp_embs = [], []
        prot_idx, comp_idx = 0, 0

        for i in range(len(batch_data['prot_emb'])):
            n_prot = len(batch_data['prot_emb'][i])
            n_comp = len(batch_data['comp_emb'][i])

            prot_embs.append(all_prot_reduced[prot_idx:prot_idx+n_prot].numpy().tolist())
            comp_embs.append(all_comp_reduced[comp_idx:comp_idx+n_comp].numpy().tolist())

            prot_idx += n_prot
            comp_idx += n_comp

        # Store results
        for i in range(len(prot_embs)):
            all_results.append({
                'prot_emb': prot_embs[i],
                'comp_emb': comp_embs[i],
                'prot_mask': batch_data['prot_mask'][i],
                'comp_mask': batch_data['comp_mask'][i],
                'label': batch_data['label'][i]
            })

        pbar.update(end_idx - start_idx)

# Save results
print("Saving dataset...")
from datasets import Dataset
result_dataset = Dataset.from_list(all_results)
result_dataset.save_to_disk(output_path)
print(f"Done! Saved to {output_path}")
