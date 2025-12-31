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

# Fit PCA on GPU (device inferred from input tensor)
print("Fitting PCA models on GPU...")
prot_pca = PCA(n_components=PROT_DIM).fit(prot_tensor)
comp_pca = PCA(n_components=COMP_DIM).fit(comp_tensor)
print(f"Protein PCA: {prot_pca.explained_variance_ratio_.sum()*100:.2f}% variance")
print(f"Compound PCA: {comp_pca.explained_variance_ratio_.sum()*100:.2f}% variance")
del prot_tensor, comp_tensor

# Transform all samples
def transform_batch(batch):
    prot_embs = [prot_pca.transform(torch.tensor(emb, dtype=torch.float32, device=device)).cpu().numpy().tolist()
                 for emb in batch['prot_emb']]
    comp_embs = [comp_pca.transform(torch.tensor(emb, dtype=torch.float32, device=device)).cpu().numpy().tolist()
                 for emb in batch['comp_emb']]
    return {'prot_emb': prot_embs, 'comp_emb': comp_embs, 'prot_mask': batch['prot_mask'],
            'comp_mask': batch['comp_mask'], 'label': batch['label']}

reduced_dataset = dataset.map(transform_batch, batched=True, batch_size=100, desc="Transforming")
reduced_dataset.save_to_disk(output_path)
print(f"Done! Saved to {output_path}")
