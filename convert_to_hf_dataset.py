#!/usr/bin/env python3
"""
Convert PyTorch cached embeddings to HuggingFace Dataset format.

Uses HuggingFace's Dataset.from_generator() with parallel processing (num_proc).
Official docs: https://huggingface.co/docs/datasets/package_reference/main_classes
"""
import os
import torch
from datasets import Dataset
import warnings

# Suppress PyTorch serialization warning
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def convert_cache_to_hf_dataset(
    cache_dir: str = "cache/embeddings",
    output_dir: str = "cache/embeddings_hf",
    num_proc: int = 4  # Disabled for debugging
):
    """Convert .pt files to HF dataset with parallel processing."""

    # Load metadata
    # metadata = torch.load(
    #     os.path.join(cache_dir, "metadata.pt"),
    #     weights_only=True  # Safe loading
    # )
    # num_samples = metadata['num_samples']
    num_samples = 143948

    print(f"Converting {num_samples:,} samples to HuggingFace Dataset")
    print(f"Running in single-process mode for debugging (num_proc={num_proc})")

    # Generator function - receives subset of indices per worker
    def sample_generator(indices):
        """Generate samples for assigned indices."""
        for idx in indices:
            sample_path = os.path.join(cache_dir, f"sample_{idx:06d}.pt")
            sample = torch.load(sample_path, weights_only=False)

            yield {
                'prot_emb': sample['prot_emb'].tolist(),
                'prot_mask': sample['prot_mask'].tolist(),
                'comp_emb': sample['comp_emb'].tolist(),
                'comp_mask': sample['comp_mask'].tolist(),
                'label': float(sample['label'])
            }

    # Create FLAT list - HuggingFace auto-splits across workers
    all_indices = list(range(num_samples))

    # Create dataset with parallel generation
    dataset = Dataset.from_generator(
        sample_generator,
        gen_kwargs={"indices": all_indices},  # FLAT list
        num_proc=num_proc
    )

    # Save
    print(f"\nSaving to {output_dir}...")
    dataset.save_to_disk(output_dir, num_proc=num_proc)

    print(f"\n✓ Done! Saved {len(dataset):,} samples")
    print(f"\nUsage: python main.py --hf-dataset {output_dir} --num-workers 16")

if __name__ == "__main__":
    convert_cache_to_hf_dataset()
