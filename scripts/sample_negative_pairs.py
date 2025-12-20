#!/usr/bin/env python3
import argparse
import os
import random
from typing import List, Set, Tuple

import pandas as pd


def _default_paths(input_path: str) -> Tuple[str, str]:
    data_dir = os.path.dirname(os.path.abspath(input_path))
    raw_path = os.path.join(data_dir, "raw.csv")
    sampled_path = os.path.join(data_dir, "sampled.csv")
    return raw_path, sampled_path


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"rna_sequence", "smiles_sequence", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "rna_sequence" in df.columns and "smiles_sequence" in df.columns:
        if "label" not in df.columns:
            df["label"] = 1
        return df[["rna_sequence", "smiles_sequence", "label"]].copy()
    if "protein_sequence" in df.columns and "smilesStructure" in df.columns:
        return pd.DataFrame(
            {
                "rna_sequence": df["protein_sequence"].fillna(""),
                "smiles_sequence": df["smilesStructure"].fillna(""),
                "label": 1,
            }
        )
    raise ValueError(
        "Input must have rna_sequence/smiles_sequence or protein_sequence/smilesStructure columns."
    )


def _sample_negative_pairs(
    rna_list: List[str],
    smiles_list: List[str],
    positive_pairs: Set[Tuple[str, str]],
    target: int,
    seed: int,
) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    negatives: Set[Tuple[str, str]] = set()
    max_attempts = target * 50 if target > 0 else 0
    attempts = 0
    while len(negatives) < target:
        if attempts > max_attempts:
            raise RuntimeError(
                "Failed to sample enough negative pairs without collisions."
            )
        rna = rng.choice(rna_list)
        smiles = rng.choice(smiles_list)
        pair = (rna, smiles)
        if pair in positive_pairs or pair in negatives:
            attempts += 1
            continue
        negatives.add(pair)
    return list(negatives)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample random negative RNA-SMILES pairs."
    )
    parser.add_argument(
        "--input",
        default="datasets/SMOPINs_openDel_3_novaSeq_SMILES_with_sequences.csv",
        help=(
            "Input CSV with protein_sequence/smilesStructure or rna_sequence/"
            "smiles_sequence columns."
        ),
    )
    parser.add_argument("--raw-output", default=None, help="Output raw CSV path.")
    parser.add_argument(
        "--sampled-output", default=None, help="Output sampled CSV path."
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    args = parser.parse_args()

    raw_path, sampled_path = _default_paths(args.input)
    if args.raw_output is None:
        args.raw_output = raw_path
    if args.sampled_output is None:
        args.sampled_output = sampled_path

    df = pd.read_csv(args.input)
    df = _normalize_columns(df)
    _validate_columns(df)

    df.to_csv(args.raw_output, index=False)

    rna_list = df["rna_sequence"].astype(str).tolist()
    smiles_list = df["smiles_sequence"].astype(str).tolist()
    unique_rna = sorted(set(rna_list))
    unique_smiles = sorted(set(smiles_list))
    positive_pairs = set(zip(rna_list, smiles_list))

    negatives = _sample_negative_pairs(
        unique_rna, unique_smiles, positive_pairs, len(df), args.seed
    )
    neg_df = pd.DataFrame(
        {
            "rna_sequence": [p[0] for p in negatives],
            "smiles_sequence": [p[1] for p in negatives],
            "label": 0,
        }
    )

    sampled = pd.concat([df, neg_df], ignore_index=True)
    sampled.to_csv(args.sampled_output, index=False)


if __name__ == "__main__":
    main()
