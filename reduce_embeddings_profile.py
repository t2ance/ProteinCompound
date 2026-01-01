#!/usr/bin/env python3
import time
import torch
from datasets import load_from_disk

# Quick profile test
dataset = load_from_disk("/home/peijia/projects/ProteinCompound/cache/embeddings_hf")
batch_data = dataset[:100]  # Get 100 samples

# Simulate the operations
print("Testing performance of each step...")

# Step 1: Flatten list comprehension
t0 = time.time()
all_prot_list = [emb for sample in batch_data['prot_emb'] for emb in sample]
t1 = time.time()
print(f"1. Flatten list comprehension: {t1-t0:.3f}s for {len(all_prot_list):,} embeddings")

# Step 2: Create tensor
t0 = time.time()
all_prot = torch.tensor(all_prot_list, dtype=torch.float32, device='cuda:0')
t1 = time.time()
print(f"2. Create GPU tensor: {t1-t0:.3f}s")

# Step 3: Dummy transform (just identity)
t0 = time.time()
all_prot_reduced = all_prot[:, :128]  # Simulate dimension reduction
t1 = time.time()
print(f"3. GPU transform: {t1-t0:.3f}s")

# Step 4: Convert to numpy
t0 = time.time()
numpy_result = all_prot_reduced.cpu().numpy()
t1 = time.time()
print(f"4. GPU->CPU + numpy: {t1-t0:.3f}s")

# Step 5: Convert to list (THE SUSPECTED BOTTLENECK)
t0 = time.time()
list_result = numpy_result.tolist()
t1 = time.time()
print(f"5. numpy->tolist(): {t1-t0:.3f}s  <-- SUSPECTED BOTTLENECK")

# Step 6: Unflatten
t0 = time.time()
prot_embs = []
idx = 0
for sample in batch_data['prot_emb']:
    n = len(sample)
    prot_embs.append(list_result[idx:idx+n])
    idx += n
t1 = time.time()
print(f"6. Unflatten: {t1-t0:.3f}s")

print(f"\nTotal time for 100 samples: {sum([t1-t0]):.3f}s")
print(f"Estimated examples/s: {100 / sum([t1-t0]):.2f}")
