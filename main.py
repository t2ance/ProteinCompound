#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from contextlib import nullcontext

def _ensure_repo_path(repo_root: str, label: str) -> None:
    if not repo_root:
        raise ValueError(f"{label} is required.")
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"{label} not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_esm(
    esm_root: str,
    model_name: str,
    checkpoint_path: Optional[str],
) -> Tuple[object, object]:
    _ensure_repo_path(esm_root, "esm_root")
    try:
        import esm
    except ImportError as exc:
        raise ImportError("Failed to import ESM; check esm_root.") from exc
    if checkpoint_path:
        print(f"loading_esm_checkpoint={checkpoint_path}", flush=True)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"esm_checkpoint not found: {checkpoint_path}")
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint_path)
        print(f"esm_loaded_from=local model={model_name}", flush=True)
        return model, alphabet
    print(f"loading_esm_model={model_name} (hub/cache)", flush=True)
    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"Unknown ESM model: {model_name}")
    model, alphabet = getattr(esm.pretrained, model_name)()
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    cache_path = os.path.join(cache_dir, f"{model_name}.pt")
    if os.path.isfile(cache_path):
        print(f"esm_checkpoint_cache={cache_path}", flush=True)
    print(
        f"esm_loaded model={model_name} params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    return model, alphabet


def _smiles_to_graph(smiles: str):
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("Missing dependency: torch-geometric.") from exc
    try:
        from dataset.smiles2graph import smiles2graph
    except ImportError as exc:
        raise ImportError("Failed to import DrugChat smiles2graph.") from exc

    graph = smiles2graph(smiles)
    return Data(
        x=torch.as_tensor(graph["node_feat"]),
        edge_index=torch.as_tensor(graph["edge_index"]),
        edge_attr=torch.as_tensor(graph["edge_feat"]),
    )


def _smiles_to_batch(smiles_list: List[str]):
    try:
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("Missing dependency: torch-geometric.") from exc
    graphs = [_smiles_to_graph(smiles) for smiles in smiles_list]
    return Batch.from_data_list(graphs)


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["rna_sequence"] = df["rna_sequence"].fillna("")
        df["smiles_sequence"] = df["smiles_sequence"].fillna("")
        df = df[(df["rna_sequence"] != "") & (df["smiles_sequence"] != "")]
        self.rna = df["rna_sequence"].astype(str).tolist()
        self.smiles = df["smiles_sequence"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.rna[idx], self.smiles[idx], self.labels[idx]


def _collate_batch(batch):
    rna, smiles, labels = zip(*batch)
    return list(rna), list(smiles), torch.tensor(labels, dtype=torch.float32)


def _binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
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


def _plot_length_distributions(
    df: pd.DataFrame,
    out_dir: str,
    wandb_run=None,
    wandb_module=None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Missing dependency: matplotlib.") from exc
    os.makedirs(out_dir, exist_ok=True)
    rna_lengths = df["rna_sequence"].astype(str).str.len()
    smiles_lengths = df["smiles_sequence"].astype(str).str.len()

    plt.figure(figsize=(6, 4))
    plt.hist(rna_lengths, bins=50, color="#4c78a8", alpha=0.85)
    plt.xlabel("RNA sequence length")
    plt.ylabel("Count")
    plt.tight_layout()
    rna_path = os.path.join(out_dir, "rna_length_hist.png")
    plt.savefig(rna_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(smiles_lengths, bins=50, color="#f58518", alpha=0.85)
    plt.xlabel("SMILES length")
    plt.ylabel("Count")
    plt.tight_layout()
    smiles_path = os.path.join(out_dir, "smiles_length_hist.png")
    plt.savefig(smiles_path, dpi=150)
    plt.close()

    if wandb_run is not None and wandb_module is not None:
        wandb_run.log(
            {
                "lengths/rna_hist": wandb_module.Histogram(
                    rna_lengths.to_numpy()
                ),
                "lengths/smiles_hist": wandb_module.Histogram(
                    smiles_lengths.to_numpy()
                ),
                "lengths/rna_plot": wandb_module.Image(rna_path),
                "lengths/smiles_plot": wandb_module.Image(smiles_path),
                "lengths/rna_mean": float(rna_lengths.mean()),
                "lengths/smiles_mean": float(smiles_lengths.mean()),
            }
        )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_mode: str,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    model.train()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    for rna, smiles, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp_mode):
            logits = model(rna, smiles, device=device)
            loss = loss_fn(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(len(loader.dataset), 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_mode: str,
) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for rna, smiles, labels in loader:
            labels = labels.to(device)
            with _amp_context(device, amp_mode):
                logits = model(rna, smiles, device=device)
                loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    if all_probs:
        probs = torch.cat(all_probs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = _binary_metrics(probs, labels)
    else:
        metrics = {}
    return total_loss / max(len(loader.dataset), 1), metrics


class ProteinEncoderESM(nn.Module):
    def __init__(
        self,
        esm_root: str,
        model_name: str,
        checkpoint_path: Optional[str],
        freeze: bool = True,
    ):
        super().__init__()
        self.model, self.alphabet = _load_esm(
            esm_root, model_name, checkpoint_path)
        print(f"ESM checkpoint loaded from {checkpoint_path}")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = self.model.num_layers
        self.embedding_dim = self.model.embed_dim
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self.freeze = freeze

    def forward(self, sequences: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.freeze:
            self.model.eval()
        data = [(str(i), seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad() if self.freeze else torch.enable_grad():
            out = self.model(tokens, repr_layers=[
                             self.repr_layer], return_contacts=False)
        reps = out["representations"][self.repr_layer]
        pad = self.alphabet.padding_idx
        lengths = tokens.ne(pad).sum(1)
        if reps.size(1) < 3:
            padding_mask = tokens.eq(pad)
            return reps, padding_mask
        reps = reps[:, 1:-1]
        valid = tokens[:, 1:-1].ne(pad)
        eos_pos = lengths - 2
        for i, pos in enumerate(eos_pos.tolist()):
            if 0 <= pos < valid.size(1):
                valid[i, pos] = False
        padding_mask = ~valid
        return reps, padding_mask


class DrugChatCompoundEncoder(nn.Module):
    def __init__(
        self,
        drugchat_root: str,
        gnn_checkpoint: str,
        num_layer: int = 5,
        emb_dim: int = 300,
        gnn_type: str = "gin",
        freeze: bool = True,
    ):
        super().__init__()
        _ensure_repo_path(drugchat_root, "drugchat_root")
        try:
            from pipeline.models.gnn import GNN_graphpred
        except ImportError as exc:
            raise ImportError(
                "Failed to import DrugChat GNN; ensure DrugChat dependencies are installed."
            ) from exc
        self.gnn = GNN_graphpred(
            num_layer,
            emb_dim,
            emb_dim,
            graph_pooling="attention",
            gnn_type=gnn_type,
        )
        if not gnn_checkpoint or not os.path.isfile(gnn_checkpoint):
            raise FileNotFoundError(
                f"gnn_checkpoint not found: {gnn_checkpoint}")
        self.gnn.from_pretrained(gnn_checkpoint)
        if freeze:
            for param in self.gnn.parameters():
                param.requires_grad = False
            self.gnn.eval()
        self.embedding_dim = emb_dim
        self.freeze = freeze

    def forward(self, smiles_list: List[str], device: torch.device) -> torch.Tensor:
        batch = _smiles_to_batch(smiles_list).to(device)
        if self.freeze:
            self.gnn.eval()
        with torch.no_grad() if self.freeze else torch.enable_grad():
            feats = self.gnn(batch)
        return feats.unsqueeze(1)


def _masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if padding_mask is None:
        return x.mean(dim=1)
    keep = (~padding_mask).unsqueeze(-1).float()
    denom = keep.sum(dim=1).clamp(min=1.0)
    return (x * keep).sum(dim=1) / denom


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _parse_list(value: str) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_lora(
    module: nn.Module,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
    bias: str,
) -> nn.Module:
    if not target_modules:
        return module
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("Missing dependency: peft.") from exc
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(module, config)


def _apply_sdpa_to_esm(model) -> None:
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("sdpa_unavailable=no_scaled_dot_product_attention", flush=True)
        return
    try:
        from esm import multihead_attention as esm_mha
    except ImportError as exc:
        raise ImportError("Failed to import ESM for SDPA.") from exc

    if getattr(esm_mha, "_sdpa_patched", False):
        return

    orig_forward = esm_mha.MultiheadAttention.forward

    def sdpa_forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        if need_head_weights:
            need_weights = True
        if (
            query.device.type != "cuda"
            or attn_mask is not None
            or before_softmax
            or incremental_state is not None
            or static_kv
            or need_weights
            or self.add_zero_attn
            or self.onnx_trace
        ):
            return orig_forward(
                self,
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_weights,
                static_kv=static_kv,
                attn_mask=attn_mask,
                before_softmax=before_softmax,
                need_head_weights=need_head_weights,
            )

        if not self.self_attention:
            return orig_forward(
                self,
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                incremental_state=incremental_state,
                need_weights=need_weights,
                static_kv=static_kv,
                attn_mask=attn_mask,
                before_softmax=before_softmax,
                need_head_weights=need_head_weights,
            )

        tgt_len, bsz, embed_dim = query.size()
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(query)
        v = self.v_proj(query)

        if self.bias_k is not None:
            bias_k = self.bias_k.repeat(1, bsz, 1)
            bias_v = self.bias_v.repeat(1, bsz, 1)
            k = torch.cat([k, bias_k], dim=0)
            v = torch.cat([v, bias_v], dim=0)
            if key_padding_mask is not None:
                pad_col = torch.zeros(
                    (bsz, 1),
                    dtype=key_padding_mask.dtype,
                    device=key_padding_mask.device,
                )
                key_padding_mask = torch.cat([key_padding_mask, pad_col], dim=1)

        src_len = k.size(0)
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        if self.rot_emb:
            q_ = q.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, tgt_len, self.head_dim)
            k_ = k.permute(0, 2, 1, 3).reshape(bsz * self.num_heads, src_len, self.head_dim)
            q_, k_ = self.rot_emb(q_, k_)
            q = q_.view(bsz, self.num_heads, tgt_len, self.head_dim).permute(0, 2, 1, 3)
            k = k_.view(bsz, self.num_heads, src_len, self.head_dim).permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

        v = v.permute(0, 2, 1, 3)

        attn_mask_sdpa = None
        if key_padding_mask is not None:
            attn_mask_sdpa = key_padding_mask.to(torch.bool).unsqueeze(1).unsqueeze(2)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_sdpa,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, embed_dim)
        attn = attn.transpose(0, 1)
        attn = self.out_proj(attn)
        return attn, None

    esm_mha.MultiheadAttention.forward = sdpa_forward
    esm_mha._sdpa_patched = True
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
    print("sdpa_patched_for_esm=True", flush=True)


def _enable_gradient_checkpointing_esm(model) -> None:
    if getattr(model, "_gc_wrapped", False):
        return
    if not hasattr(torch.utils, "checkpoint"):
        print("grad_checkpoint_unavailable=no_torch_checkpoint", flush=True)
        return

    class CheckpointedLayer(nn.Module):
        def __init__(self, layer: nn.Module):
            super().__init__()
            self.layer = layer

        def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False):
            if not self.training or need_head_weights or self_attn_mask is not None:
                return self.layer(
                    x,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_head_weights=need_head_weights,
                )

            def layer_forward(x_input):
                return self.layer(
                    x_input,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_head_weights=False,
                )

            try:
                return torch.utils.checkpoint.checkpoint(layer_forward, x, use_reentrant=False)
            except TypeError:
                return torch.utils.checkpoint.checkpoint(layer_forward, x)

    for idx, layer in enumerate(model.layers):
        if not isinstance(layer, CheckpointedLayer):
            model.layers[idx] = CheckpointedLayer(layer)
    model._gc_wrapped = True
    print("grad_checkpointing=enabled_for_esm", flush=True)


def _init_wandb(args):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Missing dependency: wandb.") from exc
    tags = _parse_list(args.wandb_tags)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=tags if tags else None,
        config=vars(args),
    )
    return run, wandb


def _amp_context(device: torch.device, amp_mode: str):
    if amp_mode == "off" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


class ProteinCompoundClassifier(nn.Module):
    def __init__(
        self,
        protein_encoder: ProteinEncoderESM,
        compound_encoder: DrugChatCompoundEncoder,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.compound_encoder = compound_encoder
        self.proj_protein = nn.Linear(
            protein_encoder.embedding_dim, hidden_dim)
        self.proj_compound = nn.Linear(
            compound_encoder.embedding_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, protein_seqs: List[str], smiles_list: List[str], device: torch.device) -> torch.Tensor:
        prot_tokens, prot_pad = self.protein_encoder(protein_seqs, device)
        comp_tokens = self.compound_encoder(smiles_list, device)
        prot_tokens = self.proj_protein(prot_tokens)
        comp_tokens = self.proj_compound(comp_tokens)
        fused, _ = self.cross_attn(prot_tokens, comp_tokens, comp_tokens)
        pooled = _masked_mean(fused, prot_pad)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def build_model(
    drugchat_root: str,
    gnn_checkpoint: str,
    esm_root: str,
    esm_model: str,
    esm_checkpoint: Optional[str],
    hidden_dim: int,
    num_heads: int,
    mlp_hidden: int,
    tuning_mode: str,
) -> ProteinCompoundClassifier:
    freeze_encoders = tuning_mode == "head"
    protein_encoder = ProteinEncoderESM(
        esm_root=esm_root,
        model_name=esm_model,
        checkpoint_path=esm_checkpoint,
        freeze=freeze_encoders,
    )
    compound_encoder = DrugChatCompoundEncoder(
        drugchat_root=drugchat_root,
        gnn_checkpoint=gnn_checkpoint,
        freeze=freeze_encoders,
    )
    return ProteinCompoundClassifier(
        protein_encoder=protein_encoder,
        compound_encoder=compound_encoder,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_hidden=mlp_hidden,
    )


def configure_tuning(
    model: ProteinCompoundClassifier,
    tuning_mode: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    lora_targets_protein: List[str],
    lora_targets_compound: List[str],
) -> None:
    if tuning_mode == "full":
        _set_requires_grad(model, True)
        return

    _set_requires_grad(model, False)
    _set_requires_grad(model.proj_protein, True)
    _set_requires_grad(model.proj_compound, True)
    _set_requires_grad(model.cross_attn, True)
    _set_requires_grad(model.classifier, True)

    if tuning_mode == "lora":
        model.protein_encoder.model = _apply_lora(
            model.protein_encoder.model,
            lora_targets_protein,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_bias,
        )
        model.compound_encoder.gnn = _apply_lora(
            model.compound_encoder.gnn,
            lora_targets_compound,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_bias,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protein-compound classifier with ESM + DrugChat.")
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
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument(
        "--tuning-mode",
        choices=["full", "lora", "head"],
        default="head",
        help="full: finetune all params; lora: LoRA on encoders + heads; head: only new modules.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-bias",
        choices=["none", "all", "lora_only"],
        default="none",
    )
    parser.add_argument(
        "--lora-targets-protein",
        default="q_proj,k_proj,v_proj",
        help="Comma-separated module names for ESM LoRA.",
    )
    parser.add_argument(
        "--lora-targets-compound",
        default="graph_pred_linear",
        help="Comma-separated module names for DrugChat GNN LoRA.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--data-csv",
        default="datasets/sampled.csv",
        help="CSV with rna_sequence, smiles_sequence, label columns.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-train", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="protein-compound")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total samples for quick experiments.",
    )
    parser.add_argument(
        "--amp",
        choices=["off", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode (CUDA only).",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Enable PyTorch SDPA for ESM (FlashAttention backend when available).",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing for ESM layers.",
    )
    args = parser.parse_args()
    wandb_run = None
    wandb_module = None
    if args.wandb:
        wandb_run, wandb_module = _init_wandb(args)
    model = build_model(
        drugchat_root=args.drugchat_root,
        gnn_checkpoint=args.gnn_checkpoint,
        esm_root=args.esm_root,
        esm_model=args.esm_model,
        esm_checkpoint=args.esm_checkpoint,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        mlp_hidden=args.mlp_hidden,
        tuning_mode=args.tuning_mode,
    )
    if args.flash_attn:
        _apply_sdpa_to_esm(model.protein_encoder.model)
    if args.grad_checkpoint:
        _enable_gradient_checkpointing_esm(model.protein_encoder.model)
    configure_tuning(
        model,
        tuning_mode=args.tuning_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_targets_protein=_parse_list(args.lora_targets_protein),
        lora_targets_compound=_parse_list(args.lora_targets_compound),
    )
    device = torch.device(args.device)
    model.to(device)
    print("#Parameters:", sum(p.numel() for p in model.parameters()))
    print(model)
    if device.type == "cpu" and args.amp != "off":
        print("amp disabled on cpu")
        args.amp = "off"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model/total_params={total_params} "
        f"model/trainable_params={trainable_params} "
        f"model/trainable_pct={100.0 * trainable_params / total_params:.2f}%"
    )

    torch.manual_seed(args.seed)
    df = pd.read_csv(args.data_csv)
    df["rna_sequence"] = df["rna_sequence"].fillna("")
    df["smiles_sequence"] = df["smiles_sequence"].fillna("")
    empty_rna = (df["rna_sequence"] == "").sum()
    empty_smiles = (df["smiles_sequence"] == "").sum()
    empty_any = ((df["rna_sequence"] == "") | (
        df["smiles_sequence"] == "")).sum()
    print(
        "dataset_rows="
        f"{len(df)} empty_rna={int(empty_rna)} "
        f"empty_smiles={int(empty_smiles)} empty_any={int(empty_any)}"
    )
    print(
        f"dataset/rows={len(df)} "
        f"dataset/empty_rna={int(empty_rna)} "
        f"dataset/empty_smiles={int(empty_smiles)} "
        f"dataset/empty_any={int(empty_any)}"
    )
    df = df[(df["rna_sequence"] != "") & (df["smiles_sequence"] != "")]
    print(f"filtered_rows={len(df)}")
    print(f"dataset/filtered_rows={len(df)}")
    label_counts = df["label"].value_counts().to_dict()
    for key, value in label_counts.items():
        print(f"dataset/label_{key}={int(value)}")

    _plot_length_distributions(
        df, os.path.dirname(args.data_csv) or ".", wandb_run, wandb_module
    )
    if args.max_samples:
        print(
            f"sampling {args.max_samples} rows from {len(df)} total rows, seed={args.seed}")
        df = df.sample(n=min(args.max_samples, len(df)),
                       random_state=args.seed)
    dataset = PairDataset(df)
    if len(dataset) < 2:
        raise RuntimeError("Dataset too small after filtering.")
    train_len = int(len(dataset) * args.train_split)
    train_len = max(min(train_len, len(dataset) - 1), 1)
    test_len = len(dataset) - train_len
    print(f"train_len={train_len} test_len={test_len}")
    generator = torch.Generator().manual_seed(args.seed)
    train_set, test_set = random_split(
        dataset, [train_len, test_len], generator=generator)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    use_fp16 = args.amp == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f'trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)} '
        f'({sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()) * 100:.2f}%)'
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model/trainable_params={trainable_params} "
        f"model/trainable_pct={100.0 * trainable_params / total_params:.2f}%"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, args.amp, scaler)
        test_loss, test_metrics = evaluate(
            model, test_loader, device, args.amp)
        metric_parts = [f"{k}={v:.4f}" for k, v in test_metrics.items()]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} {' '.join(metric_parts)}"
        )
        if wandb_run is not None:
            log_data = {
                "epoch": epoch,
                "train/loss": train_loss,
                "test/loss": test_loss,
            }
            for key, value in test_metrics.items():
                log_data[f"test/{key}"] = value
            wandb_run.log(log_data, step=epoch)
        if args.eval_train:
            train_eval_loss, train_metrics = evaluate(
                model, train_loader, device, args.amp)
            train_parts = [f"{k}={v:.4f}" for k,
                           v in train_metrics.items()]
            print(
                f"epoch={epoch} train_eval_loss={train_eval_loss:.4f} "
                f"{' '.join(train_parts)}"
            )
            if wandb_run is not None:
                log_data = {
                    "epoch": epoch,
                    "train_eval/loss": train_eval_loss,
                }
                for key, value in train_metrics.items():
                    log_data[f"train_eval/{key}"] = value
                wandb_run.log(log_data, step=epoch)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
