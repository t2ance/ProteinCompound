#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors

# Constants
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
COMPOUND_DESCRIPTORS = [
    "compound_mw",
    "compound_logp",
    "compound_heavy_atoms",
    "compound_rotatable_bonds",
    "compound_aromatic_rings",
    "compound_aliphatic_rings",
    "compound_tpsa",
    "compound_h_acceptors",
    "compound_h_donors",
    "compound_heteroatoms",
]


def _binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute binary classification metrics. Copied from main.py."""
    preds = preds.detach().cpu().float()
    labels = labels.detach().cpu().float()
    pred_labels = (preds >= 0.5).to(torch.int64)
    labels_int = labels.to(torch.int64)
    tp = int(((pred_labels == 1) & (labels_int == 1)).sum())
    tn = int(((pred_labels == 0) & (labels_int == 0)).sum())
    fp = int(((pred_labels == 1) & (labels_int == 0)).sum())
    fn = int(((pred_labels == 0) & (labels_int == 1)).sum())
    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * \
        precision * recall / (precision + recall)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None
    if roc_auc_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(
                labels_int.numpy(), preds.numpy()))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def extract_protein_features(sequence: str) -> np.ndarray:
    """
    Extract amino acid composition features from protein sequence.

    Args:
        sequence: Protein/RNA sequence string

    Returns:
        20-dimensional numpy array with amino acid frequencies
    """
    if not sequence:
        return np.zeros(20, dtype=np.float32)

    # Count each amino acid
    counts = np.zeros(20, dtype=np.float32)
    valid_count = 0

    for char in sequence.upper():
        if char in AMINO_ACIDS:
            idx = AMINO_ACIDS.index(char)
            counts[idx] += 1
            valid_count += 1

    # Normalize by total valid amino acids
    if valid_count > 0:
        counts = counts / valid_count

    return counts


def extract_compound_features(smiles: str) -> Optional[np.ndarray]:
    """
    Extract RDKit molecular descriptors from SMILES.

    Args:
        smiles: SMILES string

    Returns:
        10-dimensional numpy array with molecular descriptors, or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        features = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHeteroatoms(mol),
        ], dtype=np.float32)
        return features
    except Exception:
        return None


def extract_features_batch(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for entire dataset.

    Args:
        df: DataFrame with columns ['rna_sequence', 'smiles_sequence', 'label']

    Returns:
        Tuple of (X, y) where X is (N, 30) features and y is (N,) labels
    """
    X_list = []
    y_list = []
    invalid_count = 0

    print("Extracting features...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        protein_feats = extract_protein_features(row['rna_sequence'])
        compound_feats = extract_compound_features(row['smiles_sequence'])

        if compound_feats is None:
            invalid_count += 1
            continue

        # Concatenate protein and compound features
        combined_feats = np.concatenate([protein_feats, compound_feats])
        X_list.append(combined_feats)
        y_list.append(row['label'])

    print(f"Skipped {invalid_count} samples due to invalid SMILES")
    print(f"Using {len(X_list)} valid samples")

    if len(X_list) == 0:
        raise RuntimeError(
            "No valid samples after feature extraction. "
            "Check for invalid SMILES or empty sequences."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def load_and_filter_dataset(args) -> pd.DataFrame:
    """
    Load CSV and apply same filtering as main.py.

    Args:
        args: Command-line arguments

    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(args.data_csv)
    df["rna_sequence"] = df["rna_sequence"].fillna("")
    df["smiles_sequence"] = df["smiles_sequence"].fillna("")

    empty_rna = (df["rna_sequence"] == "").sum()
    empty_smiles = (df["smiles_sequence"] == "").sum()
    empty_any = ((df["rna_sequence"] == "") | (df["smiles_sequence"] == "")).sum()

    print(
        f"dataset_rows={len(df)} empty_rna={int(empty_rna)} "
        f"empty_smiles={int(empty_smiles)} empty_any={int(empty_any)}"
    )

    # Filter empty sequences
    df = df[(df["rna_sequence"] != "") & (df["smiles_sequence"] != "")]
    print(f"filtered_rows={len(df)}")

    # Filter by RNA sequence length
    rna_lengths = df["rna_sequence"].astype(str).str.len()
    too_long = (rna_lengths > args.max_rna_length).sum()
    print(f"max_rna_length={args.max_rna_length}")

    if too_long > 0:
        print(f"Discarding {int(too_long)} samples with RNA length > {args.max_rna_length}")
        df = df[rna_lengths <= args.max_rna_length]
        print(f"dataset/after_length_filter={len(df)}")
    else:
        print(f"All samples within max_rna_length={args.max_rna_length}")

    # Apply max_samples if specified
    if args.max_samples:
        print(f"sampling {args.max_samples} rows from {len(df)} total rows, seed={args.seed}")
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)

    return df


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, args) -> RandomForestClassifier:
    """
    Train Random Forest classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        args: Command-line arguments

    Returns:
        Trained RandomForestClassifier
    """
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.seed,
        n_jobs=-1,
        verbose=1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f}s")

    return model


def evaluate_random_forest(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate model and compute metrics.

    Args:
        model: Trained RandomForestClassifier
        X: Features
        y: Labels

    Returns:
        Dictionary of metrics
    """
    # Get predicted probabilities
    proba = model.predict_proba(X)[:, 1]

    # Convert to torch tensors for _binary_metrics compatibility
    proba_tensor = torch.tensor(proba, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Compute metrics
    metrics = _binary_metrics(proba_tensor, y_tensor)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Forest baseline for protein-compound classification."
    )

    # Data arguments
    parser.add_argument(
        "--data-csv",
        default="datasets/sampled.csv",
        help="CSV with rna_sequence, smiles_sequence, label columns.",
    )
    parser.add_argument(
        "--max-rna-length",
        type=int,
        default=4096,
        help="Maximum RNA sequence length. Longer sequences discarded.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples for quick experiments.",
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-train", action="store_true")

    # Random Forest hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees. None means unlimited.",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split an internal node.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required at a leaf node.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Number of features for best split. Options: 'sqrt', 'log2', int, float, None.",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 < args.train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    # Handle max_features argument
    if args.max_features not in ["sqrt", "log2", "None"]:
        try:
            args.max_features = int(args.max_features)
        except ValueError:
            try:
                args.max_features = float(args.max_features)
            except ValueError:
                raise ValueError("max_features must be 'sqrt', 'log2', int, float, or 'None'")
    if args.max_features == "None":
        args.max_features = None

    # Set random seed
    np.random.seed(args.seed)

    # Load and filter dataset
    df = load_and_filter_dataset(args)

    # Extract features
    X, y = extract_features_batch(df)
    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Feature distribution - mean: {X.mean():.4f}, std: {X.std():.4f}")

    if len(X) < 10:
        print(
            f"WARNING: Only {len(X)} valid samples. "
            "Results may not be reliable with such small dataset.",
            file=sys.stderr
        )

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1.0 - args.train_split,
        random_state=args.seed,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution in training set:")
    for label, count in zip(unique, counts):
        print(f"  Label {int(label)}: {int(count)} ({count/len(y_train)*100:.2f}%)")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    model = train_random_forest(X_train, y_train, args)

    # Feature importances
    importances = model.feature_importances_
    feature_names = [f"protein_{aa}" for aa in AMINO_ACIDS] + COMPOUND_DESCRIPTORS

    top_k = 10
    top_indices = np.argsort(importances)[-top_k:][::-1]

    print(f"\nTop {top_k} important features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    test_metrics = evaluate_random_forest(model, X_test, y_test)

    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Optionally evaluate on train set
    if args.eval_train:
        print("\n" + "="*50)
        print("Evaluating on train set...")
        train_metrics = evaluate_random_forest(model, X_train, y_train)

        print("\nTrain Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

    print("\n" + "="*50)
    print("Done!")


if __name__ == "__main__":
    main()
