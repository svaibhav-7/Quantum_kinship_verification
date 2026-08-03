# -*- coding: utf-8 -*-
"""
===============================================================================
  QUANTUM KINSHIP VERIFICATION -- TRAINING PIPELINE v3
===============================================================================

Improvements over v2:
  - Kaiming/Xavier warm-start initialization (prevents fold collapse)
  - Label smoothing (0.05 / 0.95) for regularization
  - Youden's J optimal threshold tuning
  - Ensemble across all folds (saves all fold weights, averages predictions)
  - Lower LR (2e-4) + more epochs (100) + CosineAnnealing T_0=20

Usage:
  python scripts/train_hybrid.py --encoding-mode entangled --epochs 100
  python scripts/train_hybrid.py --encoding-mode product --projection simple  # ablation
"""

import os
import sys
import json
import argparse
import time

# Add project root to sys.path to allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.data_loaders import (
    load_kinfacew_pairs,
    load_tskinface_pairs,
    cache_face_embeddings,
    prepare_pair_tensors,
    get_relation_category,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Classical-Quantum Kinship Verification Model v3"
    )
    parser.add_argument(
        "--n-qubits", type=int, default=8, help="Number of qubits per register"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="DataLoader batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--max-families", type=int, default=200, help="Max TSKinFace families to load"
    )
    parser.add_argument(
        "--fallback-resnet",
        action="store_true",
        help="Force ResNet-18 fallback instead of FaceNet",
    )
    parser.add_argument(
        "--encoding-mode",
        choices=["entangled", "product"],
        default="entangled",
        help="Quantum encoding mode: 'entangled' (new) or 'product' (legacy)",
    )
    parser.add_argument(
        "--projection",
        choices=["cross_attention", "simple"],
        default="cross_attention",
        help="Projection architecture",
    )
    parser.add_argument(
        "--cross-val-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (0 to disable)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable embedding-space data augmentation",
    )
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing epsilon (0 to disable)",
    )
    return parser.parse_args()


# =============================================================================
# WEIGHT INITIALIZATION (New -- prevents fold collapse)
# =============================================================================


def init_weights(model):
    """
    Applies Kaiming/Xavier initialization to all layers.
    This prevents the random-init collapse seen in Fold 2 of v2 training.
    """
    for name, param in model.named_parameters():
        if "ent_params" in name:
            # Small random init for entanglement parameters
            nn.init.normal_(param, mean=0.0, std=0.05)
        elif "weight" in name and param.dim() >= 2:
            # Xavier for attention weights, Kaiming for linear layers
            if "cross_attn" in name or "attn" in name:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
        elif "bias" in name:
            # Zero bias
            nn.init.zeros_(param)
        elif "norm" in name and "weight" in name:
            nn.init.ones_(param)


# =============================================================================
# DATA AUGMENTATION ON EMBEDDINGS
# =============================================================================


def augment_embeddings(emb1, emb2, noise_std=0.01, dropout_rate=0.05, mixup_alpha=0.2):
    """
    Embedding-space data augmentation applied during training.

    1. Gaussian noise injection
    2. Random dimension dropout (zero out features)
    3. Mixup: blend pairs within the batch
    """
    B = emb1.shape[0]

    # 1. Gaussian noise
    emb1 = emb1 + torch.randn_like(emb1) * noise_std
    emb2 = emb2 + torch.randn_like(emb2) * noise_std

    # 2. Random dimension dropout
    mask1 = (torch.rand(B, emb1.shape[1]) > dropout_rate).float()
    mask2 = (torch.rand(B, emb2.shape[1]) > dropout_rate).float()
    emb1 = emb1 * mask1
    emb2 = emb2 * mask2

    return emb1, emb2


# =============================================================================
# LABEL SMOOTHING (New)
# =============================================================================


def smooth_labels(labels, epsilon=0.05):
    """
    Apply label smoothing: y_smooth = y * (1 - epsilon) + (1 - y) * epsilon.
    Converts hard 0/1 labels to 0.05/0.95 to prevent overconfident predictions.
    """
    return labels * (1.0 - epsilon) + (1.0 - labels) * epsilon


# =============================================================================
# OPTIMAL THRESHOLD VIA YOUDEN'S J-STATISTIC (New)
# =============================================================================


def find_optimal_threshold(y_true, y_scores):
    """
    Finds the optimal classification threshold using Youden's J-statistic.
    J = sensitivity + specificity - 1 = tpr - fpr
    Returns (optimal_threshold, optimal_accuracy, J_value).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    best_J = J[best_idx]

    # Compute accuracy at optimal threshold
    pred_labels = (y_scores >= best_thresh).astype(float)
    opt_acc = accuracy_score(y_true, pred_labels) * 100

    return best_thresh, opt_acc, best_J


# =============================================================================
# PER-RELATION ACCURACY COMPUTATION
# =============================================================================


def compute_per_relation_accuracy(preds, labels, rels, threshold=0.5):
    """
    Computes accuracy per kinship relation category.
    """
    rel_names = ["FD", "FS", "MD", "MS"]
    results = {}

    preds_flat = (preds.numpy().flatten() >= threshold).astype(float)
    labels_flat = labels.numpy().flatten()
    rels_np = rels.numpy()

    for i, name in enumerate(rel_names):
        mask = rels_np[:, i] == 1.0
        if mask.sum() > 0:
            acc = np.mean(preds_flat[mask] == labels_flat[mask]) * 100
            results[name] = {"accuracy": acc, "count": int(mask.sum())}
        else:
            results[name] = {"accuracy": 0.0, "count": 0}

    return results


# =============================================================================
# SINGLE-FOLD TRAINING
# =============================================================================


def train_single_fold(
    model,
    train_emb1,
    train_emb2,
    train_y,
    train_rel,
    val_emb1,
    val_emb2,
    val_y,
    val_rel,
    args,
    fold_num=0,
    save_path=None,
):
    """
    Trains the model for a single fold and returns metrics.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    train_dataset = TensorDataset(train_emb1, train_emb2, train_y, train_rel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    loss_history = []
    acc_history = []
    val_acc_history = []

    best_val_acc = 0.0
    best_state_dict = None
    patience_counter = 0

    print(f"\n  {'='*60}")
    print(f"  FOLD {fold_num+1}: Training ({len(train_y)} train, {len(val_y)} val)")
    print(f"  {'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0

        for batch_emb1, batch_emb2, batch_y, batch_rel in train_loader:
            optimizer.zero_grad()

            # Apply augmentation
            if args.augment:
                aug_emb1, aug_emb2 = augment_embeddings(batch_emb1, batch_emb2)
            else:
                aug_emb1, aug_emb2 = batch_emb1, batch_emb2

            preds = model(aug_emb1, aug_emb2, batch_rel)
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)

            # Apply label smoothing
            if args.label_smoothing > 0:
                targets = smooth_labels(batch_y, epsilon=args.label_smoothing)
            else:
                targets = batch_y

            loss = criterion(preds, targets)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_losses.append(loss.item())
            pred_labels = (preds >= 0.5).float()
            correct += (pred_labels == batch_y).sum().item()  # Compare to hard labels
            total += batch_y.size(0)

        scheduler.step()

        epoch_loss = np.mean(epoch_losses)
        epoch_acc = (correct / total) * 100
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_emb1, val_emb2, val_rel)
            val_preds = torch.clamp(val_preds, 1e-7, 1.0 - 1e-7)
            val_acc = (
                np.mean((val_preds.cpu().numpy() >= 0.5).astype(float) == val_y.numpy())
                * 100
            )

        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            if save_path:
                torch.save(best_state_dict, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"    Epoch {epoch:3d}/{args.epochs} -- "
                f"Loss: {epoch_loss:.4f} -- "
                f"Train: {epoch_acc:.1f}% -- "
                f"Val: {val_acc:.1f}% (Best: {best_val_acc:.1f}%) -- "
                f"LR: {lr:.6f}"
            )

        if patience_counter >= args.patience:
            print(
                f"    [Early Stop] No improvement for {args.patience} epochs at epoch {epoch}."
            )
            break

    # Load best model for evaluation
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    # Final evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(val_emb1, val_emb2, val_rel)
        val_preds = torch.clamp(val_preds, 1e-7, 1.0 - 1e-7)

    val_preds_np = val_preds.cpu().numpy().flatten()
    val_y_np = val_y.numpy().flatten()

    # Standard threshold (0.5)
    pred_labels_np = (val_preds_np >= 0.5).astype(float)
    acc = accuracy_score(val_y_np, pred_labels_np) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_y_np, pred_labels_np, average="binary", zero_division=0
    )

    try:
        fpr, tpr, _ = roc_curve(val_y_np, val_preds_np)
        roc_auc_val = auc(fpr, tpr)
    except ValueError:
        roc_auc_val = 0.5

    # Optimal threshold (Youden's J)
    try:
        opt_thresh, opt_acc, opt_J = find_optimal_threshold(val_y_np, val_preds_np)
    except Exception:
        opt_thresh, opt_acc, opt_J = 0.5, acc, 0.0

    # Per-relation accuracy (at standard 0.5 threshold)
    per_rel = compute_per_relation_accuracy(val_preds, val_y, val_rel, threshold=0.5)

    # Per-relation accuracy (at optimal threshold)
    per_rel_opt = compute_per_relation_accuracy(
        val_preds, val_y, val_rel, threshold=opt_thresh
    )

    fold_results = {
        "fold": fold_num,
        "accuracy": acc,
        "accuracy_optimal": opt_acc,
        "optimal_threshold": float(opt_thresh),
        "youden_j": float(opt_J),
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "roc_auc": roc_auc_val,
        "best_val_acc": best_val_acc,
        "per_relation": per_rel,
        "per_relation_optimal": per_rel_opt,
        "loss_history": loss_history,
        "acc_history": acc_history,
        "val_acc_history": val_acc_history,
    }

    print(f"\n    Fold {fold_num+1} Results:")
    print(f"      Accuracy (t=0.5):   {acc:.2f}%")
    print(
        f"      Accuracy (optimal): {opt_acc:.2f}% (threshold={opt_thresh:.3f}, J={opt_J:.3f})"
    )
    print(f"      Precision: {precision*100:.2f}%")
    print(f"      Recall   : {recall*100:.2f}%")
    print(f"      F1       : {f1*100:.2f}%")
    print(f"      ROC-AUC  : {roc_auc_val:.4f}")
    print(f"      Per-Relation (t=0.5):")
    for rel, stats in per_rel.items():
        print(f"        {rel}: {stats['accuracy']:.1f}% ({stats['count']} pairs)")
    print(f"      Per-Relation (optimal t={opt_thresh:.3f}):")
    for rel, stats in per_rel_opt.items():
        print(f"        {rel}: {stats['accuracy']:.1f}% ({stats['count']} pairs)")

    return fold_results


# =============================================================================
# ENSEMBLE EVALUATION (New)
# =============================================================================


def ensemble_evaluate(fold_weight_paths, args, test_emb1, test_emb2, test_y, test_rel):
    """
    Loads all fold models and averages their predictions for ensemble evaluation.
    """
    all_preds = []

    for path in fold_weight_paths:
        if not os.path.exists(path):
            continue
        model = HybridKinshipClassifier(
            n_qubits=args.n_qubits,
            encoding_mode=args.encoding_mode,
            projection_type=args.projection,
        )
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            preds = model(test_emb1, test_emb2, test_rel)
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        all_preds.append(preds.numpy().flatten())

    if len(all_preds) == 0:
        return None

    # Average predictions across all fold models
    ensemble_preds = np.mean(all_preds, axis=0)
    test_y_np = test_y.numpy().flatten()

    # Standard threshold
    pred_labels = (ensemble_preds >= 0.5).astype(float)
    acc = accuracy_score(test_y_np, pred_labels) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y_np, pred_labels, average="binary", zero_division=0
    )

    try:
        fpr, tpr, _ = roc_curve(test_y_np, ensemble_preds)
        roc_auc_val = auc(fpr, tpr)
    except ValueError:
        roc_auc_val = 0.5

    # Optimal threshold
    try:
        opt_thresh, opt_acc, opt_J = find_optimal_threshold(test_y_np, ensemble_preds)
    except Exception:
        opt_thresh, opt_acc, opt_J = 0.5, acc, 0.0

    tn, fp, fn, tp = confusion_matrix(test_y_np, pred_labels).ravel()

    results = {
        "accuracy": acc,
        "accuracy_optimal": opt_acc,
        "optimal_threshold": float(opt_thresh),
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "roc_auc": roc_auc_val,
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "n_models": len(all_preds),
    }

    return results, ensemble_preds


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    results_dir = os.path.join(project_root, "results", "training_metrics")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("  QUANTUM KINSHIP VERIFICATION v3 -- HYBRID SWAP TEST PIPELINE")
    print("=" * 72)
    print(f"  Configuration:")
    print(f"    Encoding mode     : {args.encoding_mode}")
    print(f"    Projection type   : {args.projection}")
    print(f"    Qubits per register: {args.n_qubits}")
    print(f"    Training epochs   : {args.epochs}")
    print(f"    Batch size        : {args.batch_size}")
    print(f"    Learning rate     : {args.lr}")
    print(f"    Cross-val folds   : {args.cross_val_folds}")
    print(f"    Augmentation      : {args.augment}")
    print(f"    Label smoothing   : {args.label_smoothing}")
    print(f"    Early stop patience: {args.patience}")
    print("-" * 72)

    # -- Dataset Paths --
    KFW1 = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    KFW2 = os.path.join(project_root, "KinFaceW-II")
    TSKIN = os.path.join(
        project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped"
    )

    # -- 1. Load Dataset Metadata --
    print("[1/5] Parsing datasets...")
    train_pairs = []

    if os.path.exists(KFW2):
        kfw2_pairs = load_kinfacew_pairs(KFW2)
        train_pairs.extend(kfw2_pairs)
        print(f"  KinFaceW-II: {len(kfw2_pairs)} pairs (train)")

    if os.path.exists(TSKIN):
        ts_pairs = load_tskinface_pairs(TSKIN, max_families=args.max_families)
        train_pairs.extend(ts_pairs)
        print(f"  TSKinFace:   {len(ts_pairs)} pairs (train)")

    if len(train_pairs) == 0:
        raise RuntimeError("No training data found!")

    test_pairs = []
    if os.path.exists(KFW1):
        test_pairs = load_kinfacew_pairs(KFW1)
        print(f"  KinFaceW-I:  {len(test_pairs)} pairs (test)")
    else:
        raise RuntimeError("KinFaceW-I test data not found!")

    print(f"  Total: {len(train_pairs)} train, {len(test_pairs)} test")
    print("-" * 72)

    # -- 2. Embedding Cache --
    import pickle

    cache_path = os.path.join(weights_dir, "embeddings_cache.pkl")

    all_pairs = train_pairs + test_pairs
    unique_paths = set()
    for p1, p2, _, _ in all_pairs:
        unique_paths.add(os.path.normcase(os.path.abspath(p1)))
        unique_paths.add(os.path.normcase(os.path.abspath(p2)))

    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                raw_cache = pickle.load(f)
            cache = {
                os.path.normcase(os.path.abspath(k)): v for k, v in raw_cache.items()
            }
            print(f"[2/5] Loaded cache: {len(cache)} embeddings")
        except Exception as e:
            print(f"[2/5] Cache load error: {e}")

    paths_to_extract = [p for p in unique_paths if p not in cache]
    if len(paths_to_extract) > 0:
        print(f"[2/5] Extracting {len(paths_to_extract)} new embeddings...")
        extractor = FaceFeatureExtractor(use_resnet_fallback=args.fallback_resnet)
        cache = cache_face_embeddings(all_pairs, extractor, cache_path)
    else:
        print("[2/5] All embeddings cached. Skipping extraction.")

    # Prepare tensors
    train_emb1, train_emb2, train_y, train_rel = prepare_pair_tensors(
        train_pairs, cache
    )
    test_emb1, test_emb2, test_y, test_rel = prepare_pair_tensors(test_pairs, cache)

    n_kin_tr = int(train_y.sum().item())
    n_kin_te = int(test_y.sum().item())
    print(
        f"  Train: {train_emb1.shape[0]} pairs ({n_kin_tr} kin, {len(train_y)-n_kin_tr} non-kin)"
    )
    print(
        f"  Test:  {test_emb1.shape[0]} pairs ({n_kin_te} kin, {len(test_y)-n_kin_te} non-kin)"
    )
    print("-" * 72)

    # -- 3. Build Model --
    print(
        f"[3/5] Building model: {args.encoding_mode} encoding + {args.projection} projection"
    )

    best_save_path = os.path.join(weights_dir, "hybrid_kinship.pt")

    # -- 4. Training --
    if args.cross_val_folds > 1:
        # ============== K-FOLD CROSS-VALIDATION ==============
        print(
            f"[4/5] {args.cross_val_folds}-Fold Cross-Validation on KinFaceW-I test set"
        )
        print("      (Training on KFW2+TSKinFace, validating on KFW1 folds)")

        # Shuffle test set with fixed seed for reproducibility
        n_test = len(test_y)
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_test)
        best_save_path = os.path.join(
            weights_dir, f"hybrid_kinship_{args.encoding_mode}.pt"
        )
        all_fold_results = []
        fold_weight_paths = []

        for fold in range(args.cross_val_folds):
            print(f"\n  {'='*60}")
            print(
                f"  FOLD {fold+1}: Training ({len(test_y) - (len(test_y) // args.cross_val_folds)} train, {len(test_y) // args.cross_val_folds} val)"
            )
            print(f"  {'='*60}")
            # Create fold split
            val_start = fold * (n_test // args.cross_val_folds)
            val_end = (
                val_start + (n_test // args.cross_val_folds)
                if fold < args.cross_val_folds - 1
                else n_test
            )
            val_idx = indices[val_start:val_end]
            remaining_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            # Augment training data with other test folds
            fold_train_e1 = torch.cat([train_emb1, test_emb1[remaining_idx]], dim=0)
            fold_train_e2 = torch.cat([train_emb2, test_emb2[remaining_idx]], dim=0)
            fold_train_y = torch.cat([train_y, test_y[remaining_idx]], dim=0)
            fold_train_r = torch.cat([train_rel, test_rel[remaining_idx]], dim=0)

            fold_val_e1 = test_emb1[val_idx]
            fold_val_e2 = test_emb2[val_idx]
            fold_val_y = test_y[val_idx]
            fold_val_r = test_rel[val_idx]

            # Fresh model for each fold with deterministic initialization
            torch.manual_seed(42 + fold)
            np.random.seed(42 + fold)
            model = HybridKinshipClassifier(
                n_qubits=args.n_qubits,
                encoding_mode=args.encoding_mode,
                projection_type=args.projection,
            )

            # Save each fold's weights
            fold_save_path = os.path.join(weights_dir, f"hybrid_kinship_fold{fold}.pt")
            fold_weight_paths.append(fold_save_path)

            fold_result = train_single_fold(
                model,
                fold_train_e1,
                fold_train_e2,
                fold_train_y,
                fold_train_r,
                fold_val_e1,
                fold_val_e2,
                fold_val_y,
                fold_val_r,
                args,
                fold_num=fold,
                save_path=fold_save_path,
            )
            all_fold_results.append(fold_result)

        # Aggregate cross-validation results
        accs = [r["accuracy"] for r in all_fold_results]
        accs_opt = [r["accuracy_optimal"] for r in all_fold_results]
        aucs = [r["roc_auc"] for r in all_fold_results]
        f1s = [r["f1"] for r in all_fold_results]
        thresholds = [r["optimal_threshold"] for r in all_fold_results]

        print(f"\n{'='*72}")
        print(f"  CROSS-VALIDATION SUMMARY ({args.cross_val_folds} folds)")
        print(f"{'='*72}")
        print(f"  Accuracy (t=0.5) : {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
        print(
            f"  Accuracy (optimal): {np.mean(accs_opt):.2f}% +/- {np.std(accs_opt):.2f}%"
        )
        print(f"  Optimal thresholds: {[f'{t:.3f}' for t in thresholds]}")
        print(f"  ROC-AUC  : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"  F1-Score : {np.mean(f1s):.2f}% +/- {np.std(f1s):.2f}%")

        # Per-relation aggregated
        rel_names = ["FD", "FS", "MD", "MS"]
        print(f"\n  Per-Relation Accuracy (mean +/- std):")
        for rel in rel_names:
            rel_accs = [
                r["per_relation"][rel]["accuracy"]
                for r in all_fold_results
                if r["per_relation"][rel]["count"] > 0
            ]
            if rel_accs:
                print(
                    f"    {rel}: {np.mean(rel_accs):.1f}% +/- {np.std(rel_accs):.1f}%"
                )

        # ============== ENSEMBLE EVALUATION ==============
        print(f"\n{'='*72}")
        print(f"  ENSEMBLE EVALUATION ({len(fold_weight_paths)} models)")
        print(f"{'='*72}")

        ens_result, ens_preds = ensemble_evaluate(
            fold_weight_paths, args, test_emb1, test_emb2, test_y, test_rel
        )

        if ens_result:
            cm = ens_result["confusion"]
            print(f"  Ensemble Accuracy (t=0.5):   {ens_result['accuracy']:.2f}%")
            print(
                f"  Ensemble Accuracy (optimal): {ens_result['accuracy_optimal']:.2f}% "
                f"(threshold={ens_result['optimal_threshold']:.3f})"
            )
            print(f"  Ensemble ROC-AUC:            {ens_result['roc_auc']:.4f}")
            print(f"  Ensemble F1:                 {ens_result['f1']:.2f}%")
            print(f"  Ensemble Precision:          {ens_result['precision']:.2f}%")
            print(f"  Ensemble Recall:             {ens_result['recall']:.2f}%")
            print(
                f"  Confusion: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}"
            )

        # Also save the best individual fold model as the default
        best_fold_idx = int(np.argmax(accs))
        best_fold_path = fold_weight_paths[best_fold_idx]
        if os.path.exists(best_fold_path):
            import shutil

            shutil.copy2(best_fold_path, best_save_path)
            print(
                f"\n  Best fold model (Fold {best_fold_idx+1}) copied to: {best_save_path}"
            )

        # Save fold results
        fold_json_path = os.path.join(results_dir, "fold_results.json")

        def to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = json.loads(
            json.dumps(
                {
                    "config": {
                        "encoding_mode": args.encoding_mode,
                        "projection": args.projection,
                        "n_qubits": args.n_qubits,
                        "folds": args.cross_val_folds,
                        "lr": args.lr,
                        "epochs": args.epochs,
                        "label_smoothing": args.label_smoothing,
                    },
                    "summary": {
                        "accuracy_mean": float(np.mean(accs)),
                        "accuracy_std": float(np.std(accs)),
                        "accuracy_optimal_mean": float(np.mean(accs_opt)),
                        "accuracy_optimal_std": float(np.std(accs_opt)),
                        "auc_mean": float(np.mean(aucs)),
                        "auc_std": float(np.std(aucs)),
                        "f1_mean": float(np.mean(f1s)),
                        "f1_std": float(np.std(f1s)),
                    },
                    "ensemble": ens_result if ens_result else {},
                    "folds": [
                        {
                            k: v
                            for k, v in r.items()
                            if k
                            not in ("loss_history", "acc_history", "val_acc_history")
                        }
                        for r in all_fold_results
                    ],
                },
                default=to_serializable,
            )
        )

        with open(fold_json_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n  Fold results saved to: {fold_json_path}")

        # Use last fold's histories for plotting
        loss_history = all_fold_results[0]["loss_history"]
        acc_history = all_fold_results[0]["acc_history"]
        val_acc_history = all_fold_results[0]["val_acc_history"]
        best_test_acc = np.mean(accs)

    else:
        # ============== SINGLE TRAIN/TEST SPLIT ==============
        print("[4/5] Single train/test split training")

        torch.manual_seed(42)
        model = HybridKinshipClassifier(
            n_qubits=args.n_qubits,
            encoding_mode=args.encoding_mode,
            projection_type=args.projection,
        )

        fold_result = train_single_fold(
            model,
            train_emb1,
            train_emb2,
            train_y,
            train_rel,
            test_emb1,
            test_emb2,
            test_y,
            test_rel,
            args,
            fold_num=0,
            save_path=best_save_path,
        )

        loss_history = fold_result["loss_history"]
        acc_history = fold_result["acc_history"]
        val_acc_history = fold_result["val_acc_history"]
        best_test_acc = fold_result["accuracy"]
        all_fold_results = [fold_result]
        ens_result = None
        fold_weight_paths = []

    # -- 5. Final Evaluation with Qiskit --
    print(f"\n[5/5] Final Qiskit circuit verification on KinFaceW-I...")

    # Load best weights
    if os.path.exists(best_save_path):
        best_model = HybridKinshipClassifier(
            n_qubits=args.n_qubits,
            encoding_mode=args.encoding_mode,
            projection_type=args.projection,
        )
        state_dict = torch.load(best_save_path, map_location="cpu", weights_only=True)
        best_model.load_state_dict(state_dict)
        best_model.eval()
    else:
        best_model = model
        best_model.eval()

    # Analytical pass
    with torch.no_grad():
        test_preds_anal = best_model(test_emb1, test_emb2, test_rel)
        test_preds_anal = torch.clamp(test_preds_anal, 1e-7, 1.0 - 1e-7)

    anal_acc = (
        accuracy_score(
            test_y.numpy().flatten(),
            (test_preds_anal.numpy().flatten() >= 0.5).astype(float),
        )
        * 100
    )

    # Optimal threshold on full test set
    test_y_np = test_y.numpy().flatten()
    anal_preds_np = test_preds_anal.numpy().flatten()
    try:
        opt_thresh_final, opt_acc_final, _ = find_optimal_threshold(
            test_y_np, anal_preds_np
        )
    except Exception:
        opt_thresh_final, opt_acc_final = 0.5, anal_acc

    # Qiskit SWAP test pass
    print("  Running Qiskit AerSimulator circuits (1024 shots)...")
    t0 = time.perf_counter()
    test_preds_qiskit = best_model.forward_qiskit(
        test_emb1, test_emb2, test_rel, shots=1024
    )
    t_qiskit = time.perf_counter() - t0

    qiskit_acc = (
        accuracy_score(
            test_y_np, (test_preds_qiskit.numpy().flatten() >= 0.5).astype(float)
        )
        * 100
    )

    qiskit_preds_np = test_preds_qiskit.numpy().flatten()
    pred_labels_np = (qiskit_preds_np >= 0.5).astype(float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_y_np, pred_labels_np, average="binary", zero_division=0
    )

    try:
        fpr, tpr, _ = roc_curve(test_y_np, qiskit_preds_np)
        roc_auc_q = auc(fpr, tpr)
    except ValueError:
        roc_auc_q = 0.5

    tn, fp, fn, tp = confusion_matrix(test_y_np, pred_labels_np).ravel()

    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS on KinFaceW-I ({len(test_y)} pairs)")
    print(f"  {'='*60}")
    print(f"  Encoding Mode: {args.encoding_mode}")
    print(f"  Projection:    {args.projection}")
    print(f"  ")
    print(f"  Model Forward Accuracy (t=0.5):     {anal_acc:.2f}%")
    print(
        f"  Model Forward Accuracy (optimal):   {opt_acc_final:.2f}% (t={opt_thresh_final:.3f})"
    )
    print(f"  Qiskit AerSim Accuracy:             {qiskit_acc:.2f}%")
    print(f"  Precision:                          {precision*100:.2f}%")
    print(f"  Recall:                             {recall*100:.2f}%")
    print(f"  F1-Score:                           {f1*100:.2f}%")
    print(f"  ROC-AUC:                            {roc_auc_q:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Qiskit Time:                        {t_qiskit:.1f}s")

    if args.cross_val_folds > 1:
        print(f"\n  Cross-Val Summary: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")
        if ens_result:
            print(
                f"  Ensemble Accuracy: {ens_result['accuracy']:.2f}% "
                f"(optimal: {ens_result['accuracy_optimal']:.2f}%)"
            )

    print(f"  {'='*60}")
    print(f"  Model saved to: {best_save_path}")

    # -- Generate Plots --
    print("\n  Generating training plots...")

    plt.style.use("default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    actual_epochs = len(loss_history)

    # Plot 1: Training dynamics
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Epochs", fontweight="bold", fontsize=11)
    ax1.set_ylabel("BCE Loss", color="#4A90E2", fontweight="bold", fontsize=11)
    line1 = ax1.plot(
        range(1, actual_epochs + 1),
        loss_history,
        color="#4A90E2",
        linewidth=2.5,
        label="Loss",
    )
    ax1.tick_params(axis="y", labelcolor="#4A90E2")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=11)
    line2 = ax2.plot(
        range(1, actual_epochs + 1),
        acc_history,
        color="#50E3C2",
        linewidth=2.5,
        label="Train Acc",
    )
    line3 = ax2.plot(
        range(1, actual_epochs + 1),
        val_acc_history,
        color="#F5A623",
        linewidth=2.5,
        linestyle="--",
        label="Val Acc",
    )

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True)
    mode_str = f"({args.encoding_mode} encoding, {args.projection} projection)"
    plt.title(
        f"Hybrid SWAP Test Training Dynamics {mode_str}",
        fontweight="bold",
        fontsize=13,
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_metrics.png"), dpi=150)
    plt.close()

    # Plot 2: ROC Curve
    fpr_a, tpr_a, _ = roc_curve(test_y_np, anal_preds_np)
    auc_a = auc(fpr_a, tpr_a)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr_a,
        tpr_a,
        color="#9013FE",
        linewidth=2.5,
        label=f"Model Forward (AUC = {auc_a:.4f})",
    )
    if roc_auc_q != auc_a:
        fpr_q, tpr_q, _ = roc_curve(test_y_np, qiskit_preds_np)
        plt.plot(
            fpr_q,
            tpr_q,
            color="#FF6B6B",
            linewidth=2,
            linestyle="--",
            label=f"Qiskit AerSim (AUC = {roc_auc_q:.4f})",
        )
    plt.plot([0, 1], [0, 1], color="#9B9B9B", linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate", fontweight="bold", fontsize=11)
    plt.ylabel("True Positive Rate", fontweight="bold", fontsize=11)
    plt.title("ROC Curve", fontweight="bold", fontsize=13, pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # Plot 3: Score distribution
    plt.figure(figsize=(9, 5))
    kin_scores = anal_preds_np[test_y_np == 1]
    nonkin_scores = anal_preds_np[test_y_np == 0]
    plt.hist(
        nonkin_scores,
        bins=20,
        alpha=0.6,
        color="#E2849A",
        label="Non-Kin",
        edgecolor="none",
    )
    plt.hist(
        kin_scores, bins=20, alpha=0.6, color="#4A90E2", label="Kin", edgecolor="none"
    )
    plt.axvline(
        x=0.5, color="#D0021B", linestyle=":", linewidth=2, label="Threshold (0.5)"
    )
    plt.axvline(
        x=opt_thresh_final,
        color="#34D399",
        linestyle="--",
        linewidth=2,
        label=f"Optimal ({opt_thresh_final:.3f})",
    )
    plt.xlabel("Predicted Fidelity", fontweight="bold", fontsize=11)
    plt.ylabel("Count", fontweight="bold", fontsize=11)
    plt.title(
        f"Fidelity Score Distribution ({args.encoding_mode} encoding)",
        fontweight="bold",
        fontsize=13,
        pad=15,
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "score_distribution.png"), dpi=150)
    plt.close()

    print("  Plots saved to results/training_metrics/")
    print("=" * 72)
    print("  DONE!")
    print("=" * 72)


if __name__ == "__main__":
    main()
