#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import (
    PairDataset,
    ProteinEncoderESM,
    DrugChatCompoundEncoder,
    _collate_batch,
)


def scan_existing_cache(output_dir: str, prefix: str = "sample") -> set:
    """
    Scan output directory and return set of already-cached sample indices.

    Returns:
        Set of integers representing cached sample indices
    """
    if not os.path.exists(output_dir):
        return set()

    existing_indices = set()
    pattern = f"{prefix}_"

    for filename in os.listdir(output_dir):
        if filename.startswith(pattern) and filename.endswith(".pt"):
            try:
                idx_str = filename[len(pattern):-3]
                idx = int(idx_str)
                existing_indices.add(idx)
            except ValueError:
                continue

    return existing_indices


def extract_and_save_embeddings(
    protein_encoder,
    compound_encoder,
    dataset,
    device: torch.device,
    output_dir: str,
    batch_size: int = 8,
    prefix: str = "sample",
):
    os.makedirs(output_dir, exist_ok=True)

    existing_indices = scan_existing_cache(output_dir, prefix)
    if existing_indices:
        print(f"Found {len(existing_indices)} cached samples, will skip them")
        print(f"Cached indices range: {min(existing_indices)} to {max(existing_indices)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch
    )

    protein_encoder.eval()
    compound_encoder.eval()

    sample_idx = 0
    skipped_count = 0
    saved_count = 0

    with torch.no_grad():
        for rna_seqs, smiles_list, labels in tqdm(loader, desc="Extracting embeddings"):
            batch_indices = list(range(sample_idx, sample_idx + len(rna_seqs)))
            uncached_positions = [i for i, idx in enumerate(batch_indices) if idx not in existing_indices]

            if not uncached_positions:
                skipped_count += len(rna_seqs)
                sample_idx += len(rna_seqs)
                continue

            prot_tokens, prot_pad = protein_encoder(rna_seqs, device)
            comp_tokens, comp_pad = compound_encoder(smiles_list, device)

            batch_size_actual = len(rna_seqs)
            for i in range(batch_size_actual):
                current_idx = sample_idx + i

                if current_idx in existing_indices:
                    skipped_count += 1
                    continue

                prot_len = (~prot_pad[i]).sum().item()
                comp_len = (~comp_pad[i]).sum().item()

                sample_data = {
                    'prot_emb': prot_tokens[i, :prot_len].cpu(),
                    'prot_mask': prot_pad[i, :prot_len].cpu(),
                    'comp_emb': comp_tokens[i, :comp_len].cpu(),
                    'comp_mask': comp_pad[i, :comp_len].cpu(),
                    'label': labels[i].item(),
                    'rna_sequence': rna_seqs[i],
                    'smiles_sequence': smiles_list[i],
                }

                output_path = os.path.join(output_dir, f"{prefix}_{current_idx:06d}.pt")
                torch.save(sample_data, output_path)
                saved_count += 1

            sample_idx += batch_size_actual

    print(f"Extraction complete!")
    print(f"  Total samples: {sample_idx}")
    print(f"  Newly saved: {saved_count}")
    print(f"  Skipped (cached): {skipped_count}")

    metadata = {
        'num_samples': sample_idx,
        'protein_encoder_dim': prot_tokens.size(-1),
        'compound_encoder_dim': comp_tokens.size(-1),
    }
    metadata_path = os.path.join(output_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache protein/compound embeddings for training."
    )
    parser.add_argument(
        "--data-csv",
        required=True,
        help="CSV with rna_sequence, smiles_sequence, label columns."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save extracted embeddings."
    )
    parser.add_argument(
        "--drugchat-root",
        default=os.environ.get("DRUGCHAT_ROOT", "external/drugchat"),
    )
    parser.add_argument(
        "--gnn-checkpoint",
        default=os.environ.get(
            "DRUGCHAT_GNN_CKPT",
            "external/drugchat/ckpt/gin_contextpred.pth",
        ),
    )
    parser.add_argument(
        "--esm-root",
        default=os.environ.get("ESM_ROOT", "external/esm"),
    )
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--esm-checkpoint", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-rna-length",
        type=int,
        default=4096,
        help="Maximum RNA sequence length."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples for quick experiments."
    )
    parser.add_argument("--seed", type=int, default=13)

    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    df["rna_sequence"] = df["rna_sequence"].fillna("")
    df["smiles_sequence"] = df["smiles_sequence"].fillna("")

    empty_rna = (df["rna_sequence"] == "").sum()
    empty_smiles = (df["smiles_sequence"] == "").sum()
    print(f"Dataset rows: {len(df)}, empty_rna: {empty_rna}, empty_smiles: {empty_smiles}")

    df = df[(df["rna_sequence"] != "") & (df["smiles_sequence"] != "")]

    rna_lengths = df["rna_sequence"].astype(str).str.len()
    too_long = (rna_lengths > args.max_rna_length).sum()
    if too_long > 0:
        print(f"Discarding {int(too_long)} samples with RNA length > {args.max_rna_length}")
        df = df[rna_lengths <= args.max_rna_length]

    if args.max_samples:
        print(f"Sampling {args.max_samples} rows from {len(df)} total rows")
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)

    print(f"Shuffling dataset with seed={args.seed}")
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    dataset = PairDataset(df)
    print(f"Dataset size after filtering: {len(dataset)}")

    print("Building encoders...")
    protein_encoder = ProteinEncoderESM(
        esm_root=args.esm_root,
        model_name=args.esm_model,
        checkpoint_path=args.esm_checkpoint,
        freeze=True,
    )
    print(f"Protein encoder: ESM {args.esm_model}")

    compound_encoder = DrugChatCompoundEncoder(
        drugchat_root=args.drugchat_root,
        gnn_checkpoint=args.gnn_checkpoint,
        freeze=True,
    )
    print(f"Compound encoder: DrugChat GNN")

    protein_encoder.to(device)
    compound_encoder.to(device)

    print(f"Extracting embeddings to {args.output_dir}")
    extract_and_save_embeddings(
        protein_encoder=protein_encoder,
        compound_encoder=compound_encoder,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        prefix="sample",
    )

    print("Extraction complete!")


if __name__ == "__main__":
    main()
