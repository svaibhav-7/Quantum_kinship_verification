# -*- coding: utf-8 -*-
"""
Improved FIW retraining script for the kinship ensemble.

This version is designed to generalize better than the older retraining script by:
1. using a strict family-level train/validation split,
2. fine-tuning from the older checkpoint instead of training from scratch,
3. using class-weighted BCE loss to handle kin/non-kin imbalance,
4. early stopping on the validation set rather than on CV folds.

Usage:
    python scripts/training/train_with_fiw_improved.py
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_fscore_support, roc_curve


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loaders import cache_face_embeddings, prepare_pair_tensors
from src.models_improved import FaceFeatureExtractor, HybridKinshipClassifier
from scripts.training.train_with_fiw import load_fiw_pairs_split


def split_families(families, val_ratio=0.2, seed=42):
    """Split families into train/validation sets while keeping them disjoint."""
    if len(families) < 3:
        raise ValueError("Need at least 3 families for a train/validation split")
    rng = np.random.default_rng(seed)
    shuffled = list(families)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_families = set(shuffled[:val_size])
    train_families = set(shuffled[val_size:])
    return sorted(train_families), sorted(val_families)


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        target = target.float()
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        weights = target * self.pos_weight + (1.0 - target)
        return (weights * bce).mean()


def build_model(base_checkpoint=None, n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention"):
    model = HybridKinshipClassifier(
        n_qubits=n_qubits,
        encoding_mode=encoding_mode,
        projection_type=projection_type,
    )
    if base_checkpoint and os.path.exists(base_checkpoint):
        state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Some old checkpoints may be wrapped or use a different key layout.
            model.load_state_dict(state, strict=False)
    return model


def train_model(model, train_loader, val_loader, args, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = WeightedBCELoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_state = None
    best_val_auc = -1.0
    best_val_acc = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for emb1, emb2, y, rel in train_loader:
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            y = y.to(device).float().view(-1, 1)
            rel = rel.to(device)

            optimizer.zero_grad()
            preds = model(emb1, emb2, rel)
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        val_scores = []
        val_labels = []
        with torch.no_grad():
            for emb1, emb2, y, rel in val_loader:
                emb1 = emb1.to(device)
                emb2 = emb2.to(device)
                rel = rel.to(device)
                scores = model(emb1, emb2, rel).cpu().numpy().flatten()
                val_scores.extend(scores)
                val_labels.extend(y.numpy().astype(float).flatten())

        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)
        val_preds = (val_scores >= 0.5).astype(float)
        val_acc = accuracy_score(val_labels, val_preds) * 100.0
        fpr, tpr, _ = roc_curve(val_labels, val_scores)
        val_auc = auc(fpr, tpr)

        scheduler.step(val_auc)
        history.append((epoch, float(np.mean(epoch_losses)), val_acc, val_auc))

        improved = val_auc > best_val_auc + args.improvement_threshold or (
            abs(val_auc - best_val_auc) <= args.improvement_threshold and val_acc > best_val_acc
        )
        if improved:
            best_val_auc = float(val_auc)
            best_val_acc = float(val_acc)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}: loss={np.mean(epoch_losses):.4f}, val_acc={val_acc:.2f}%, val_auc={val_auc:.4f}")

        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(f"  Auto-stopping at epoch {epoch}: validation has not improved for {args.patience} checks")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_acc": best_val_acc, "best_val_auc": best_val_auc}


def main():
    parser = argparse.ArgumentParser(description="Improved FIW retraining for kinship verification")
    parser.add_argument("--fiw-root", type=str, default=os.path.join(ROOT, "public"))
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs; the script will stop earlier if validation stops improving")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-epochs", type=int, default=20, help="Minimum epochs to run before early stopping can activate")
    parser.add_argument("--patience", type=int, default=12, help="Stop if validation does not improve for this many checks after min-epochs")
    parser.add_argument("--improvement-threshold", type=float, default=1e-4, help="Minimum validation improvement needed to count as progress")
    parser.add_argument("--pos-weight", type=float, default=2.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-checkpoint", type=str, default=os.path.join(ROOT, "weights", "hybrid_kinship_improved_entangled.pt"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(ROOT, "outputs", "fiw_retraining_improved"), help="Directory for checkpoints, cache, and metrics")
    parser.add_argument("--save-path", type=str, default=None, help="Optional explicit checkpoint path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_path is None:
        args.save_path = os.path.join(args.output_dir, "best_checkpoint.pt")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print("[1/4] Loading FIW family split...")
    train_pairs, val_pairs, split_info = load_fiw_pairs_split(args.fiw_root, train_ratio=args.train_ratio, seed=args.seed)
    print(f"  train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")

    print("[2/4] Extracting embeddings...")
    extractor = FaceFeatureExtractor()
    cache_path = os.path.join(args.output_dir, "fiw_improved_cache.pkl")
    cache = cache_face_embeddings(train_pairs + val_pairs, extractor, cache_path)

    train_emb1, train_emb2, train_y, train_rel = prepare_pair_tensors(train_pairs, cache)
    val_emb1, val_emb2, val_y, val_rel = prepare_pair_tensors(val_pairs, cache)

    train_dataset = TensorDataset(train_emb1, train_emb2, train_y, train_rel)
    val_dataset = TensorDataset(val_emb1, val_emb2, val_y, val_rel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("[3/4] Fine-tuning from older checkpoint...")
    model = build_model(base_checkpoint=args.base_checkpoint)
    model, stats = train_model(model, train_loader, val_loader, args, args.save_path)

    print("[4/4] Evaluating best checkpoint...")
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for emb1, emb2, y, rel in val_loader:
            pred = model(emb1, emb2, rel).cpu().numpy().flatten()
            scores.extend(pred)
            labels.extend(y.numpy().astype(float).flatten())

    scores = np.array(scores)
    labels = np.array(labels)
    preds = (scores >= 0.5).astype(float)
    acc = accuracy_score(labels, preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    print(f"  Validation accuracy: {acc:.2f}%")
    print(f"  Validation precision: {precision * 100:.2f}%")
    print(f"  Validation recall: {recall * 100:.2f}%")
    print(f"  Validation F1: {f1 * 100:.2f}%")
    print(f"  Validation AUC: {auc_score:.4f}")

    metrics_path = os.path.join(args.output_dir, "validation_metrics.json")
    metrics = {
        "val_accuracy": float(acc),
        "val_precision": float(precision * 100.0),
        "val_recall": float(recall * 100.0),
        "val_f1": float(f1 * 100.0),
        "val_auc": float(auc_score),
        "split_info": split_info,
        "stats": stats,
    }
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    print(f"  Saved improved checkpoint to: {args.save_path}")
    print(f"  Saved metrics to: {metrics_path}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
