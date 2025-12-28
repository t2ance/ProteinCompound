#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath("external/esm"))
sys.path.insert(0, os.path.abspath("external/drugchat/dataset"))
sys.path.insert(0, os.path.abspath("external/drugchat"))

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext
import wandb
from tqdm import tqdm
import esm
from esm import multihead_attention as esm_mha
from torch_geometric.data import Data, Batch
from dataset.smiles2graph import smiles2graph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pipeline.models.gnn import GNN_graphpred
from peft import inject_adapter_in_model, LoraConfig
from datasets import load_from_disk

import main_ml
def ensure_repo_path(repo_root: str, label: str) -> None:
    if not repo_root:
        raise ValueError(f"{label} is required.")
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"{label} not found: {repo_root}")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_esm(
    esm_root: str,
    model_name: str,
    checkpoint_path: Optional[str],
) -> Tuple[object, object]:
    ensure_repo_path(esm_root, "esm_root")
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


def smiles_to_graph(smiles: str):
    graph = smiles2graph(smiles)
    return Data(
        x=torch.as_tensor(graph["node_feat"]),
        edge_index=torch.as_tensor(graph["edge_index"]),
        edge_attr=torch.as_tensor(graph["edge_feat"]),
    )


def smiles_to_batch(smiles_list: List[str]):
    graphs = [smiles_to_graph(smiles) for smiles in smiles_list]
    return Batch.from_data_list(graphs)


def collate_hf_batch(batch):
    """Collate function for HuggingFace dataset format (list of dicts)."""
    prot_embs = [sample['prot_emb'] for sample in batch]
    prot_masks = [sample['prot_mask'] for sample in batch]
    comp_embs = [sample['comp_emb'] for sample in batch]
    comp_masks = [sample['comp_mask'] for sample in batch]
    labels = [sample['label'] for sample in batch]

    max_prot_len = max(e.size(0) for e in prot_embs)
    max_comp_len = max(e.size(0) for e in comp_embs)

    prot_dim = prot_embs[0].size(1)
    comp_dim = comp_embs[0].size(1)
    batch_size = len(batch)

    padded_prot = torch.zeros(batch_size, max_prot_len, prot_dim)
    padded_prot_mask = torch.ones(batch_size, max_prot_len, dtype=torch.bool)
    padded_comp = torch.zeros(batch_size, max_comp_len, comp_dim)
    padded_comp_mask = torch.ones(batch_size, max_comp_len, dtype=torch.bool)

    for i, (prot_emb, prot_mask, comp_emb, comp_mask) in enumerate(zip(prot_embs, prot_masks, comp_embs, comp_masks)):
        prot_len = prot_emb.size(0)
        comp_len = comp_emb.size(0)

        padded_prot[i, :prot_len] = prot_emb
        padded_prot_mask[i, :prot_len] = prot_mask
        padded_comp[i, :comp_len] = comp_emb
        padded_comp_mask[i, :comp_len] = comp_mask

    labels_tensor = torch.stack(labels)

    return padded_prot, padded_prot_mask, padded_comp, padded_comp_mask, labels_tensor


def binary_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    # Convert to numpy arrays
    preds_np = preds.detach().cpu().float().numpy()
    labels_np = labels.detach().cpu().float().numpy()
    pred_labels_np = (preds_np >= 0.5).astype(int)
    labels_int_np = labels_np.astype(int)

    metrics = {
        "accuracy": float(accuracy_score(labels_int_np, pred_labels_np)),
        "precision": float(precision_score(labels_int_np, pred_labels_np, zero_division=0)),
        "recall": float(recall_score(labels_int_np, pred_labels_np, zero_division=0)),
        "f1": float(f1_score(labels_int_np, pred_labels_np, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_int_np, preds_np)),
    }

    return metrics

def train_step(
    model: nn.Module,
    batch_data: Tuple,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_mode: str,
    scaler: Optional[torch.cuda.amp.GradScaler],
    accumulation_steps: int,
    step_within_accumulation: int,
    grad_clip: Optional[float] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_precomputed_embeddings: bool = True,
) -> Tuple[float, Optional[float]]:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    if use_precomputed_embeddings:
        prot_emb, prot_mask, comp_emb, comp_mask, labels = batch_data
        labels = labels.to(device)

        with amp_context(device, amp_mode):
            logits = model(prot_emb, prot_mask, comp_emb, comp_mask, device)
            loss = loss_fn(logits, labels)
    else:
        rna, smiles, labels = batch_data
        labels = labels.to(device)

        with amp_context(device, amp_mode):
            logits = model(rna, smiles, device=device)
            loss = loss_fn(logits, labels)

    # Scale loss for gradient accumulation
    scaled_loss = loss / accumulation_steps

    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    # Only update optimizer every N steps
    is_accumulation_boundary = (step_within_accumulation == accumulation_steps - 1)
    grad_norm = None

    if is_accumulation_boundary:
        # Unscale gradients first if using GradScaler
        if scaler is not None:
            scaler.unscale_(optimizer)

        # Apply gradient clipping if specified and compute norm
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            ).item()
        else:
            # Compute gradient norm without clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Step the scheduler after each optimizer update
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)

    return loss.item(), grad_norm


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_mode: str,
    use_precomputed_embeddings: bool = False,
) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_data in tqdm(loader, total=len(loader), desc="Evaluating"):

            if use_precomputed_embeddings:
                prot_emb, prot_mask, comp_emb, comp_mask, labels = batch_data
                labels = labels.to(device)

                with amp_context(device, amp_mode):
                    logits = model(prot_emb, prot_mask, comp_emb, comp_mask, device)
                    loss = loss_fn(logits, labels)
            else:
                rna, smiles, labels = batch_data
                labels = labels.to(device)

                with amp_context(device, amp_mode):
                    logits = model(rna, smiles, device=device)
                    loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    if all_probs:
        probs = torch.cat(all_probs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        metrics = binary_metrics(probs, labels)
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
        self.model, self.alphabet = load_esm(
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
        if self.freeze:
            with torch.no_grad():
                out = self.model(tokens, repr_layers=[
                                 self.repr_layer], return_contacts=False)
        else:
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


class ManualProteinEncoder(nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.embedding_dim = len(main_ml.AMINO_ACIDS)
        self.normalize = normalize
        self.register_buffer(
            "feature_mean",
            torch.zeros(self.embedding_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_std",
            torch.ones(self.embedding_dim, dtype=torch.float32),
        )
        self.freeze = True

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.embedding_dim or std.numel() != self.embedding_dim:
            raise ValueError("Manual protein normalization stats have wrong size.")
        mean = mean.to(self.feature_mean.device, dtype=torch.float32)
        std = std.to(self.feature_std.device, dtype=torch.float32)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def enable_gradient_checkpointing(self):
        return

    def forward(self, sequences: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        invalid_mask = []
        for sequence in sequences:
            feats = main_ml.extract_protein_features(sequence)
            if np.isclose(feats.sum(), 0.0):
                features.append(np.zeros(self.embedding_dim, dtype=np.float32))
                invalid_mask.append(True)
            else:
                features.append(feats)
                invalid_mask.append(False)
        feats_np = np.stack(features, axis=0).astype(np.float32)
        feats_tensor = torch.from_numpy(feats_np).to(
            device=device, dtype=torch.float32
        )
        if self.normalize:
            mean = self.feature_mean.to(device=device, dtype=feats_tensor.dtype)
            std = self.feature_std.to(device=device, dtype=feats_tensor.dtype)
            feats_tensor = (feats_tensor - mean) / std
            if any(invalid_mask):
                invalid = torch.tensor(invalid_mask, device=device)
                feats_tensor[invalid] = 0.0
        feats_tensor = feats_tensor.unsqueeze(1)
        padding_mask = torch.zeros(
            feats_tensor.size(0), 1, dtype=torch.bool, device=device
        )
        return feats_tensor, padding_mask


class ManualCompoundEncoder(nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.embedding_dim = len(main_ml.COMPOUND_DESCRIPTORS)
        self.normalize = normalize
        self.register_buffer(
            "feature_mean",
            torch.zeros(self.embedding_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_std",
            torch.ones(self.embedding_dim, dtype=torch.float32),
        )
        self.freeze = True

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.embedding_dim or std.numel() != self.embedding_dim:
            raise ValueError("Manual compound normalization stats have wrong size.")
        mean = mean.to(self.feature_mean.device, dtype=torch.float32)
        std = std.to(self.feature_std.device, dtype=torch.float32)
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def enable_gradient_checkpointing(self):
        return

    def forward(self, smiles_list: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        invalid_mask = []
        for smiles in smiles_list:
            feats = main_ml.extract_compound_features(smiles)
            if feats is None:
                features.append(np.zeros(self.embedding_dim, dtype=np.float32))
                invalid_mask.append(True)
            else:
                features.append(feats)
                invalid_mask.append(False)
        feats_np = np.stack(features, axis=0).astype(np.float32)
        feats_tensor = torch.from_numpy(feats_np).to(
            device=device, dtype=torch.float32
        )
        if self.normalize:
            mean = self.feature_mean.to(device=device, dtype=feats_tensor.dtype)
            std = self.feature_std.to(device=device, dtype=feats_tensor.dtype)
            feats_tensor = (feats_tensor - mean) / std
            if any(invalid_mask):
                invalid = torch.tensor(invalid_mask, device=device)
                feats_tensor[invalid] = 0.0
        feats_tensor = feats_tensor.unsqueeze(1)
        padding_mask = torch.zeros(
            feats_tensor.size(0), 1, dtype=torch.bool, device=device
        )
        return feats_tensor, padding_mask


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
        ensure_repo_path(drugchat_root, "drugchat_root")
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

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for GNN to save memory."""
        if getattr(self.gnn, "_gc_wrapped", False):
            return

        original_forward = self.gnn.forward

        def checkpointed_forward(x, edge_index, edge_attr):
            if not self.gnn.training:
                return original_forward(x, edge_index, edge_attr)

            def forward_fn(x_input, edge_index_input, edge_attr_input):
                return original_forward(x_input, edge_index_input, edge_attr_input)

            return torch.utils.checkpoint.checkpoint(
                forward_fn,
                x, edge_index, edge_attr,
                use_reentrant=False
            )

        self.gnn.forward = checkpointed_forward
        self.gnn._gc_wrapped = True
        print("grad_checkpointing=enabled_for_gnn", flush=True)

    def forward(self, smiles_list: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = smiles_to_batch(smiles_list).to(device)
        if self.freeze:
            self.gnn.eval()

        # Get node-level features (before pooling)
        if self.freeze:
            with torch.no_grad():
                node_feats = self.gnn.gnn(batch.x, batch.edge_index, batch.edge_attr)
        else:
            node_feats = self.gnn.gnn(batch.x, batch.edge_index, batch.edge_attr)

        # Convert to batched format with padding
        batch_size = batch.batch.max().item() + 1
        max_nodes = max((batch.batch == i).sum().item() for i in range(batch_size))

        # Create padded tensor [batch, max_nodes, emb_dim] using node_feats device
        padded_feats = torch.zeros(
            batch_size, max_nodes, node_feats.size(1),
            device=node_feats.device,  # Use node_feats.device directly
            dtype=node_feats.dtype,
            pin_memory=False
        )
        padding_mask = torch.ones(
            batch_size, max_nodes,
            device=node_feats.device,
            dtype=torch.bool,
            pin_memory=False
        )

        # Fill in node features per graph using in-place operations
        node_idx = 0
        for b in range(batch_size):
            num_nodes = (batch.batch == b).sum().item()
            # Use narrow + copy_ for more efficient in-place operation
            padded_feats[b].narrow(0, 0, num_nodes).copy_(
                node_feats[node_idx:node_idx + num_nodes]
            )
            padding_mask[b].narrow(0, 0, num_nodes).fill_(False)  # False = not padded
            node_idx += num_nodes

        return padded_feats, padding_mask


def masked_mean(x: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if padding_mask is None:
        return x.mean(dim=1)
    keep = (~padding_mask).unsqueeze(-1).float()
    denom = keep.sum(dim=1).clamp(min=1.0)
    return (x * keep).sum(dim=1) / denom


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def parse_list(value: str) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_lora(
    module: nn.Module,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
    bias: str,
) -> nn.Module:
    if not target_modules:
        return module
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
    )

    inject_adapter_in_model(config, module)

    for name, submodule in module.named_modules():
        if hasattr(submodule, 'enable_adapters'):
            submodule.enable_adapters(True)
        if hasattr(submodule, 'set_adapter'):
            submodule.set_adapter('default')

    for name, param in module.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    return module


def apply_sdpa_to_esm(model) -> None:
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("sdpa_unavailable=no_scaled_dot_product_attention", flush=True)
        return

    if getattr(esm_mha, "_sdpa_patched", False):
        return

    orig_forward = esm_mha.MultiheadAttention.forward

    def _should_use_sdpa(self, query, key, value, key_padding_mask, incremental_state,
                         need_weights, static_kv, attn_mask, before_softmax):
        """Check if we can use SDPA for this forward pass."""
        return (
            query.device.type == "cuda"
            and self.self_attention
            and attn_mask is None
            and not before_softmax
            and incremental_state is None
            and not static_kv
            and not need_weights
            and not self.add_zero_attn
            and not self.onnx_trace
            and self.bias_k is None  # Simplify: skip bias_k/bias_v cases
            # Removed: and not self.rot_emb  # Now supports rotary embeddings!
        )

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

        # Fast path: use SDPA for simple self-attention cases
        if _should_use_sdpa(self, query, key, value, key_padding_mask, incremental_state,
                           need_weights, static_kv, attn_mask, before_softmax):
            tgt_len, bsz, embed_dim = query.size()

            # Project Q, K, V
            q = self.q_proj(query) * self.scaling
            k = self.k_proj(query)
            v = self.v_proj(query)

            # Reshape to [batch*heads, seq_len, head_dim] for rotary embeddings
            q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            # Apply rotary embeddings if present (ESM2 uses this)
            if self.rot_emb:
                q, k = self.rot_emb(q, k)

            # Reshape to [batch, num_heads, seq_len, head_dim] for SDPA
            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, tgt_len, self.head_dim)
            v = v.view(bsz, self.num_heads, tgt_len, self.head_dim)

            # Convert padding mask to attention mask format for SDPA
            attn_mask_sdpa = None
            if key_padding_mask is not None:
                attn_mask_sdpa = key_padding_mask.unsqueeze(1).unsqueeze(2)

            # Use SDPA with FlashAttention backend
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_sdpa,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

            # Reshape back to [seq_len, batch, embed_dim]
            attn = attn.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
            attn = self.out_proj(attn)
            return attn, None

        # Fallback to original implementation for complex cases
        return orig_forward(
            self, query, key, value,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_weights,
            static_kv=static_kv,
            attn_mask=attn_mask,
            before_softmax=before_softmax,
            need_head_weights=need_head_weights,
        )

    esm_mha.MultiheadAttention.forward = sdpa_forward
    esm_mha._sdpa_patched = True

    # Enable flash attention backends if available
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    print("sdpa_patched_for_esm=True (with_rotary_embedding_support)", flush=True)

    # Log available SDPA backends for verification
    if torch.cuda.is_available():
        try:
            flash_enabled = torch.backends.cuda.flash_sdp_enabled()
            mem_efficient_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
            math_enabled = torch.backends.cuda.math_sdp_enabled()
            print(f"sdpa_backends: flash={flash_enabled}, mem_efficient={mem_efficient_enabled}, math={math_enabled}", flush=True)
        except Exception:
            pass


def enable_gradient_checkpointing_esm(model) -> None:
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


def enable_gradient_checkpointing_cross_attn(model) -> None:
    """Enable gradient checkpointing for cross-attention module."""
    if getattr(model, "_cross_attn_gc_wrapped", False):
        return

    # Skip if using concat fusion mode (no cross_attn)
    if model.cross_attn is None:
        return

    original_forward = model.cross_attn.forward

    def checkpointed_cross_attn(query, key, value, key_padding_mask=None,
                                need_weights=True, attn_mask=None,
                                average_attn_weights=True):
        if not model.training:
            return original_forward(query, key, value,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=need_weights,
                                   attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights)

        # Checkpointed path - force need_weights=False to save memory
        def forward_fn(q, k, v):
            out, _ = original_forward(q, k, v,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=False,
                                     attn_mask=attn_mask,
                                     average_attn_weights=average_attn_weights)
            return out

        result = torch.utils.checkpoint.checkpoint(
            forward_fn, query, key, value, use_reentrant=False
        )
        return result, None

    model.cross_attn.forward = checkpointed_cross_attn
    model._cross_attn_gc_wrapped = True
    print("grad_checkpointing=enabled_for_cross_attention", flush=True)


def amp_context(device: torch.device, amp_mode: str):
    if amp_mode == "off" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


class LightweightProteinCompoundClassifier(nn.Module):
    """Lightweight classifier for precomputed embeddings (no encoders)."""
    def __init__(
        self,
        prot_emb_dim: int = 1280,
        comp_emb_dim: int = 300,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
        fusion_mode: str = "xattn",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.prot_emb_dim = prot_emb_dim
        self.comp_emb_dim = comp_emb_dim

        self.proj_protein = nn.Linear(prot_emb_dim, hidden_dim)
        self.proj_compound = nn.Linear(comp_emb_dim, hidden_dim)

        if fusion_mode == "xattn":
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True)
            classifier_input_dim = hidden_dim
        elif fusion_mode == "concat":
            self.cross_attn = None
            classifier_input_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        prot_tokens: torch.Tensor,
        prot_pad: torch.Tensor,
        comp_tokens: torch.Tensor,
        comp_pad: torch.Tensor,
        device: torch.device,
        debug: bool = False
    ) -> torch.Tensor:
        prot_tokens = prot_tokens.to(device, non_blocking=True)
        prot_pad = prot_pad.to(device, non_blocking=True)
        comp_tokens = comp_tokens.to(device, non_blocking=True)
        comp_pad = comp_pad.to(device, non_blocking=True)

        prot_tokens = self.proj_protein(prot_tokens)
        comp_tokens = self.proj_compound(comp_tokens)

        if self.fusion_mode == "xattn":
            fused, _ = self.cross_attn(
                prot_tokens, comp_tokens, comp_tokens,
                key_padding_mask=comp_pad,
                need_weights=False,
                average_attn_weights=True
            )
            pooled = masked_mean(fused, prot_pad)
        elif self.fusion_mode == "concat":
            pooled_prot = masked_mean(prot_tokens, prot_pad)
            pooled_comp = masked_mean(comp_tokens, comp_pad)
            pooled = torch.cat([pooled_prot, pooled_comp], dim=-1)

        logits = self.classifier(pooled).squeeze(-1)
        return logits


class ProteinCompoundClassifier(nn.Module):
    def __init__(
        self,
        protein_encoder: nn.Module,
        compound_encoder: nn.Module,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
        fusion_mode: str = "xattn",
    ):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.compound_encoder = compound_encoder
        self.fusion_mode = fusion_mode

        self.proj_protein = nn.Linear(
            protein_encoder.embedding_dim, hidden_dim)
        self.proj_compound = nn.Linear(
            compound_encoder.embedding_dim, hidden_dim)

        # Cross-attention only for xattn mode
        if fusion_mode == "xattn":
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True)
            classifier_input_dim = hidden_dim
        elif fusion_mode == "concat":
            self.cross_attn = None
            classifier_input_dim = hidden_dim * 2  # concat protein + compound
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, protein_seqs: List[str], smiles_list: List[str], device: torch.device, debug: bool = False) -> torch.Tensor:
        # Get encoder outputs with masks
        prot_tokens, prot_pad = self.protein_encoder(protein_seqs, device)  # [B, L_p, D_p]
        comp_tokens, comp_pad = self.compound_encoder(smiles_list, device)  # [B, L_c, D_c]

        if debug:
            print(f"\n[ENCODER OUTPUTS]")
            print(f"  prot_tokens: {prot_tokens.shape}, mean={prot_tokens.mean().item():.4f}, std={prot_tokens.std().item():.4f}")
            print(f"  prot_pad: {prot_pad.shape}, True%={prot_pad.float().mean().item():.2%} (True=padding)")
            print(f"  comp_tokens: {comp_tokens.shape}, mean={comp_tokens.mean().item():.4f}, std={comp_tokens.std().item():.4f}")
            print(f"  comp_pad: {comp_pad.shape}, True%={comp_pad.float().mean().item():.2%} (True=padding)")

        # Project to common dimension
        prot_tokens = self.proj_protein(prot_tokens)  # [B, L_p, 512]
        comp_tokens = self.proj_compound(comp_tokens)  # [B, L_c, 512]

        if debug:
            print(f"\n[AFTER PROJECTION]")
            print(f"  prot_tokens: {prot_tokens.shape}, mean={prot_tokens.mean().item():.4f}, std={prot_tokens.std().item():.4f}")
            print(f"  comp_tokens: {comp_tokens.shape}, mean={comp_tokens.mean().item():.4f}, std={comp_tokens.std().item():.4f}")

        # Fusion: either cross-attention or concat
        if self.fusion_mode == "xattn":
            # Cross-attention: protein (Q) attends to compound atoms (K, V)
            fused, attn_weights = self.cross_attn(
                prot_tokens,              # query
                comp_tokens,              # key
                comp_tokens,              # value
                key_padding_mask=comp_pad, # mask padded compound atoms
                need_weights=False,        # Disable for memory efficiency
                average_attn_weights=True
            )

            if debug:
                print(f"\n[AFTER CROSS-ATTENTION]")
                print(f"  fused: {fused.shape}, mean={fused.mean().item():.4f}, std={fused.std().item():.4f}")
                if attn_weights is not None:
                    print(f"  attn_weights: {attn_weights.shape}, mean={attn_weights.mean().item():.4f}, std={attn_weights.std().item():.4f}")
                    print(f"  attn_weights per query: min={attn_weights.min(dim=-1)[0].mean().item():.4f}, max={attn_weights.max(dim=-1)[0].mean().item():.4f}")
                    # Check if attention is uniform or peaked
                    entropy = -(attn_weights * (attn_weights + 1e-10).log()).sum(dim=-1).mean()
                    max_entropy = torch.log(torch.tensor(attn_weights.size(-1), dtype=torch.float32))
                    print(f"  attention entropy: {entropy.item():.4f} / {max_entropy.item():.4f} (higher=more uniform)")

            # Pool fused protein tokens (respecting protein padding)
            pooled = masked_mean(fused, prot_pad)

        elif self.fusion_mode == "concat":
            # Simple concatenation: pool both then concat
            pooled_prot = masked_mean(prot_tokens, prot_pad)  # [B, 512]
            pooled_comp = masked_mean(comp_tokens, comp_pad)  # [B, 512]
            pooled = torch.cat([pooled_prot, pooled_comp], dim=-1)  # [B, 1024]

            if debug:
                print(f"\n[AFTER CONCAT FUSION]")
                print(f"  pooled_prot: {pooled_prot.shape}, mean={pooled_prot.mean().item():.4f}, std={pooled_prot.std().item():.4f}")
                print(f"  pooled_comp: {pooled_comp.shape}, mean={pooled_comp.mean().item():.4f}, std={pooled_comp.std().item():.4f}")
                print(f"  pooled: {pooled.shape}, mean={pooled.mean().item():.4f}, std={pooled.std().item():.4f}")

        if debug:
            print(f"\n[AFTER POOLING]")
            print(f"  pooled: {pooled.shape}, mean={pooled.mean().item():.4f}, std={pooled.std().item():.4f}")

        # Classify
        logits = self.classifier(pooled).squeeze(-1)

        if debug:
            print(f"\n[CLASSIFIER OUTPUT]")
            print(f"  logits: {logits.shape}, mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

        return logits

    def forward_from_embeddings(
        self,
        prot_tokens: torch.Tensor,
        prot_pad: torch.Tensor,
        comp_tokens: torch.Tensor,
        comp_pad: torch.Tensor,
        device: torch.device,
        debug: bool = False
    ) -> torch.Tensor:
        prot_tokens = prot_tokens.to(device, non_blocking=True)
        prot_pad = prot_pad.to(device, non_blocking=True)
        comp_tokens = comp_tokens.to(device, non_blocking=True)
        comp_pad = comp_pad.to(device, non_blocking=True)

        prot_tokens = self.proj_protein(prot_tokens)
        comp_tokens = self.proj_compound(comp_tokens)

        if self.fusion_mode == "xattn":
            fused, _ = self.cross_attn(
                prot_tokens, comp_tokens, comp_tokens,
                key_padding_mask=comp_pad,
                need_weights=False,
                average_attn_weights=True
            )
            pooled = masked_mean(fused, prot_pad)
        elif self.fusion_mode == "concat":
            pooled_prot = masked_mean(prot_tokens, prot_pad)
            pooled_comp = masked_mean(comp_tokens, comp_pad)
            pooled = torch.cat([pooled_prot, pooled_comp], dim=-1)

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
    fusion_mode: str = "xattn",
    use_precomputed_embeddings: bool = True,
    manual_compound_features: bool = False,
    manual_protein_features: bool = False,
) -> nn.Module:
    if use_precomputed_embeddings:
        return LightweightProteinCompoundClassifier(
            prot_emb_dim=1280,
            comp_emb_dim=300,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_hidden=mlp_hidden,
            fusion_mode=fusion_mode,
        )
    else:
        freeze_encoders = tuning_mode == "head"
        if manual_protein_features:
            protein_encoder = ManualProteinEncoder(normalize=True)
        else:
            protein_encoder = ProteinEncoderESM(
                esm_root=esm_root,
                model_name=esm_model,
                checkpoint_path=esm_checkpoint,
                freeze=freeze_encoders,
            )
        if manual_compound_features:
            compound_encoder = ManualCompoundEncoder(normalize=True)
        else:
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
            fusion_mode=fusion_mode,
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
        set_requires_grad(model, True)
        if hasattr(model.protein_encoder, "freeze"):
            model.protein_encoder.freeze = False
        if hasattr(model.compound_encoder, "freeze"):
            model.compound_encoder.freeze = False
        return

    set_requires_grad(model, False)
    set_requires_grad(model.proj_protein, True)
    set_requires_grad(model.proj_compound, True)
    if model.cross_attn is not None:
        set_requires_grad(model.cross_attn, True)
    set_requires_grad(model.classifier, True)

    if tuning_mode == "lora":
        if hasattr(model.protein_encoder, "freeze"):
            model.protein_encoder.freeze = False
        if hasattr(model.compound_encoder, "freeze"):
            model.compound_encoder.freeze = False
        if hasattr(model.protein_encoder, "model"):
            model.protein_encoder.model = apply_lora(
                model.protein_encoder.model,
                lora_targets_protein,
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_bias,
            )
        if hasattr(model.compound_encoder, "gnn"):
            model.compound_encoder.gnn = apply_lora(
                model.compound_encoder.gnn,
                lora_targets_compound,
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_bias,
            )


def load_hf_datasets(
    hf_dataset_path: str,
    train_split: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Load datasets from HuggingFace dataset directory using best practices.

    Returns:
        train_set, test_set (HuggingFace Dataset instances)
    """
    print(f"Using HuggingFace dataset: {hf_dataset_path}")
    if not os.path.exists(hf_dataset_path):
        raise FileNotFoundError(f"HuggingFace dataset not found: {hf_dataset_path}")

    full_dataset = load_from_disk(hf_dataset_path)
    num_cached_samples = len(full_dataset)
    print(f"Found {num_cached_samples} cached samples in HF dataset")

    all_indices = list(range(num_cached_samples))
    random.seed(seed)
    random.shuffle(all_indices)

    train_len = int(num_cached_samples * train_split)
    train_len = max(min(train_len, num_cached_samples - 1), 1)
    test_len = num_cached_samples - train_len

    train_indices = all_indices[:train_len]
    test_indices = all_indices[train_len:]

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]
        print(f"Limiting training samples to {len(train_indices)} (max_train_samples={max_train_samples})", flush=True)

    if max_val_samples is not None:
        test_indices = test_indices[:max_val_samples]
        print(f"Limiting validation samples to {len(test_indices)} (max_val_samples={max_val_samples})", flush=True)

    print(f"train_len={len(train_indices)} test_len={len(test_indices)}")

    train_set = full_dataset.select(train_indices)
    test_set = full_dataset.select(test_indices)

    # Convert HF dataset to return PyTorch tensors (not lists)
    train_set.set_format(
        type='torch',
        columns=['prot_emb', 'prot_mask', 'comp_emb', 'comp_mask', 'label']
    )
    test_set.set_format(
        type='torch',
        columns=['prot_emb', 'prot_mask', 'comp_emb', 'comp_mask', 'label']
    )

    return train_set, test_set


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
        "--num-steps",
        type=int,
        required=True,
        help="Total number of training steps (optimizer updates, not batches)."
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer type. 'adamw': AdamW optimizer, 'sgd': SGD with momentum.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer (ignored for AdamW).")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--log-steps",
        type=int,
        default=10,
        help="Log training metrics (batch loss, grad norms, LR) to wandb every N steps."
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Run validation every N steps. If None, defaults to num_steps // 10 (10%% of training)."
    )
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
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples (default: None, no limit).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=1000,
        help="Limit validation samples (default: 1000).",
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
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before optimizer step.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1,
        help="Gradient clipping threshold (max norm). If None, no clipping is applied.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Ratio of total steps for linear warmup (default: 0.06 = 6%%).",
    )
    parser.add_argument(
        "--scheduler-type",
        choices=["none", "linear_warmup_cosine", "cosine"],
        default="linear_warmup_cosine",
        help="Learning rate scheduler type. 'none': no scheduler, 'linear_warmup_cosine': warmup + decay, 'cosine': cosine annealing only.",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=["concat", "xattn"],
        default="concat",
        help="Fusion mode for combining protein and compound representations. 'concat': simple concatenation + MLP, 'xattn': cross-attention (original).",
    )
    parser.add_argument(
        "--hf-dataset",
        required=True,
        help="Path to HuggingFace dataset directory (created by convert_to_hf_dataset.py).",
    )
    parser.add_argument(
        "--use-precomputed-embeddings",
        action="store_true",
        default=True,
        help="Use precomputed embeddings from HF dataset (lightweight model). If False, compute embeddings on-the-fly (full model with encoders).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of worker processes for data loading. 0 means main process only (slow). Higher values improve GPU utilization.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=8,
        help="Number of batches to prefetch per worker (only used if num_workers > 0).",
    )
    args = parser.parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    print(f"Dataset limits: max_train_samples={args.max_train_samples if args.max_train_samples is not None else 'None (no limit)'}, max_val_samples={args.max_val_samples}", flush=True)

    tags = parse_list(args.wandb_tags)
    wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            tags=tags if tags else None,
            config=vars(args),
        )

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
        fusion_mode=args.fusion_mode,
        use_precomputed_embeddings=args.use_precomputed_embeddings,
        manual_compound_features=False,
        manual_protein_features=False,
    )
    if args.use_precomputed_embeddings:
        print(
            f"model=LightweightProteinCompoundClassifier "
            f"(no encoders, precomputed embeddings) "
            f"fusion_mode={args.fusion_mode}",
            flush=True,
        )
    else:
        print(
            "protein_encoder=esm "
            f"model={args.esm_model} checkpoint={args.esm_checkpoint}",
            flush=True,
        )
        print(
            f"compound_encoder=drugchat gnn_checkpoint={args.gnn_checkpoint}",
            flush=True,
        )
        print(f"fusion_mode={args.fusion_mode}", flush=True)
    if not args.use_precomputed_embeddings:
        if args.flash_attn:
            apply_sdpa_to_esm(model.protein_encoder.model)
        if args.grad_checkpoint:
            enable_gradient_checkpointing_esm(model.protein_encoder.model)
            model.compound_encoder.enable_gradient_checkpointing()
            enable_gradient_checkpointing_cross_attn(model)
    if not args.use_precomputed_embeddings:
        configure_tuning(
            model,
            tuning_mode=args.tuning_mode,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_bias=args.lora_bias,
            lora_targets_protein=parse_list(args.lora_targets_protein),
            lora_targets_compound=parse_list(args.lora_targets_compound),
        )
    else:
        print("model=lightweight tuning_mode=all_trainable", flush=True)
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

    # Load HuggingFace datasets
    train_set, test_set = load_hf_datasets(
        hf_dataset_path=args.hf_dataset,
        train_split=args.train_split,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # Use HuggingFace collate function (precomputed embeddings)
    collate_fn = collate_hf_batch
    use_precomputed_embeddings = args.use_precomputed_embeddings

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    print(f"dataloader/num_workers={args.num_workers} "
          f"dataloader/prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'N/A'} "
          f"dataloader/pin_memory={True if device.type == 'cuda' else False} "
          f"dataloader/persistent_workers={True if args.num_workers > 0 else False}", flush=True)

    # Total training steps is directly specified by user
    total_training_steps = args.num_steps
    print(f"scheduler/total_training_steps={total_training_steps}", flush=True)

    use_fp16 = args.amp == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # Create optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(f"optimizer/type=AdamW lr={args.lr} weight_decay={args.weight_decay}", flush=True)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        print(f"optimizer/type=SGD lr={args.lr} momentum={args.momentum} weight_decay={args.weight_decay}", flush=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
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

    # Create learning rate scheduler
    scheduler = None
    if args.scheduler_type == "linear_warmup_cosine":
        warmup_steps = max(1, int(args.warmup_ratio * total_training_steps))
        print(f"scheduler/type=linear_warmup_cosine scheduler/warmup_steps={warmup_steps} "
              f"scheduler/warmup_ratio={args.warmup_ratio:.4f}", flush=True)

        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-10/args.lr, end_factor=1.0, total_iters=warmup_steps
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_training_steps - warmup_steps, eta_min=0
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps]
        )
    elif args.scheduler_type == "cosine":
        print(f"scheduler/type=cosine scheduler/T_max={total_training_steps}", flush=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_training_steps, eta_min=0
        )
    else:
        print("scheduler/type=none", flush=True)

    # Step-based training loop
    global_step = 0
    max_steps = args.num_steps

    # Smart default for eval_steps if not specified
    eval_steps = args.eval_steps
    if eval_steps is None:
        eval_steps = max(1, max_steps // 10)
        print(f"eval_steps not specified, using smart default: {eval_steps} (10% of num_steps)", flush=True)

    # Create infinite iterator over train_loader
    train_iterator = iter(train_loader)
    step_within_accumulation = 0

    # Initialize optimizer state
    optimizer.zero_grad(set_to_none=True)

    # Progress bar
    pbar = tqdm(total=max_steps, desc="Training")

    print(f'Using precomputed embeddings: {use_precomputed_embeddings}', flush=True)

    while global_step < max_steps:
        # Get next batch (wrap around if needed)
        try:
            batch_data = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_data = next(train_iterator)

        # Single training step
        loss, grad_norm = train_step(
            model=model,
            batch_data=batch_data,
            optimizer=optimizer,
            device=device,
            amp_mode=args.amp,
            scaler=scaler,
            accumulation_steps=args.gradient_accumulation_steps,
            step_within_accumulation=step_within_accumulation,
            grad_clip=args.grad_clip,
            scheduler=scheduler,
            use_precomputed_embeddings=use_precomputed_embeddings,
        )

        # Update accumulation counter
        step_within_accumulation = (step_within_accumulation + 1) % args.gradient_accumulation_steps

        # Only increment global_step on optimizer update boundaries
        if step_within_accumulation == 0:
            global_step += 1
            pbar.update(1)

            # Logging every log_steps
            if global_step % args.log_steps == 0:
                log_data = {
                    "step": global_step,
                    "train/batch_loss": loss,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                }
                # Add gradient norm if available (only on accumulation boundaries)
                if grad_norm is not None:
                    log_data["train/grad_norm"] = grad_norm
                wandb.log(log_data)

            # Validation every eval_steps
            if global_step % eval_steps == 0:
                val_loss, val_metrics = evaluate(
                    model, test_loader, device, args.amp,
                    use_precomputed_embeddings=use_precomputed_embeddings
                )

                metric_parts = [f"{k}={v:.4f}" for k, v in val_metrics.items()]
                print(f"\nstep={global_step} val_loss={val_loss:.4f} {' '.join(metric_parts)}")

                log_data = {
                    "step": global_step,
                    "val/loss": val_loss,
                }
                for key, value in val_metrics.items():
                    log_data[f"val/{key}"] = value

                # Add memory metrics
                if device.type == "cuda":
                    log_data["memory/allocated_gb"] = torch.cuda.memory_allocated(device) / 1024**3
                    log_data["memory/reserved_gb"] = torch.cuda.memory_reserved(device) / 1024**3
                wandb.log(log_data)

                model.train()  # Switch back to training mode

    pbar.close()

    wandb.finish()


if __name__ == "__main__":
    main()
