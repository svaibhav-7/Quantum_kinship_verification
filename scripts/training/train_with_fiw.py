# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- RETRAIN WITH FIW 50% + EXISTING DATA
=================================================================================

Splits FIW at the family level into 50% train / 50% test.
Combines FIW-train with KinFaceW-II + TSKinFace for training.
Trains a 5-fold cross-validation ensemble and evaluates on the held-out FIW 50%.

Usage:
  python scripts/train_with_fiw.py
  python scripts/train_with_fiw.py --epochs 120 --patience 25
"""

import os
import sys
import csv
import glob
import json
import time
import pickle
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Robust project root resolution (works whether run from root or any scripts subfolder)
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "training", "evaluation", "inference", "archive"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

def find_weight_file(filename, project_root):
    """Search for weight/cache file across weights subdirectories."""
    search_dirs = [
        os.path.join(project_root, "weights", "caches"),
        os.path.join(project_root, "weights", "active_ensemble"),
        os.path.join(project_root, "weights", "folds"),
        os.path.join(project_root, "weights"),
        os.path.join(project_root, "weights", "archive"),
    ]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(project_root, "weights", filename)

from src.models_improved import (
    FaceFeatureExtractor,
    HybridKinshipClassifier,
    EnsembleKinshipClassifier,
)
from src.data_loaders import (
    load_kinfacew_pairs,
    load_tskinface_pairs,
    cache_face_embeddings,
    prepare_pair_tensors,
    get_relation_category,
)


# =============================================================================
# FIW Data Loading (family-level split)
# =============================================================================

def get_mid_images(fpath, mid_id):
    """Find all image files in MID folder regardless of folder/file casing or extension."""
    target_mid = f"mid{mid_id}".lower()
    mid_dir = None
    if os.path.exists(fpath):
        for d in os.listdir(fpath):
            d_clean = d.lower()
            try:
                if d_clean == target_mid or d_clean == f"mid{int(mid_id)}":
                    mid_dir = os.path.join(fpath, d)
                    break
            except ValueError:
                if d_clean == target_mid:
                    mid_dir = os.path.join(fpath, d)
                    break
    if not mid_dir or not os.path.exists(mid_dir):
        return []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for f in os.listdir(mid_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            images.append(os.path.join(mid_dir, f))
    return images


def load_fiw_pairs_split(fiw_root, train_ratio=0.5, seed=42):
    """
    Load FIW dataset and split at the FAMILY level into train/test sets.

    Returns:
        train_pairs: list of (img1, img2, label, rel_str) for training families
        test_pairs:  list of (img1, img2, label, rel_str) for testing families
        split_info:  dict with split statistics
    """
    fids_dir = os.path.join(fiw_root, "FIDs") if os.path.exists(os.path.join(fiw_root, "FIDs")) else fiw_root
    if not os.path.exists(fids_dir):
        print(f"  [ERROR] FIDs directory not found at {fids_dir}")
        return [], [], {}

    families = sorted([d for d in os.listdir(fids_dir) if os.path.isdir(os.path.join(fids_dir, d))])
    print(f"  [FIW] Found {len(families)} families in {fids_dir}")

    # Split families 50/50
    rng = random.Random(seed)
    shuffled_families = list(families)
    rng.shuffle(shuffled_families)
    split_idx = int(len(shuffled_families) * train_ratio)
    train_families = set(shuffled_families[:split_idx])
    test_families = set(shuffled_families[split_idx:])

    print(f"  [FIW] Split: {len(train_families)} train families, {len(test_families)} test families")

    # Parse all families and collect kin pairs per family
    family_kin_pairs = {}  # fid -> list of kin pairs
    all_family_images = {}  # fid -> list of image paths

    for fid in families:
        fpath = os.path.join(fids_dir, fid)
        mid_csv = os.path.join(fpath, "mid.csv")
        if not os.path.exists(mid_csv):
            # Try case-insensitive lookup for mid.csv
            for item in os.listdir(fpath) if os.path.exists(fpath) else []:
                if item.lower() == "mid.csv":
                    mid_csv = os.path.join(fpath, item)
                    break
        if not os.path.exists(mid_csv):
            continue

        with open(mid_csv, "r") as f:
            reader = list(csv.reader(f))
        if len(reader) < 2:
            continue

        header = [x.strip() for x in reader[0]]
        mids = []
        genders = {}
        rel_matrix = {}
        all_family_images[fid] = []

        for row in reader[1:]:
            if not row:
                continue
            mid_id = row[0].strip()
            mids.append(mid_id)
            gender = row[-1].strip().lower()
            genders[mid_id] = gender
            rel_matrix[mid_id] = {}
            for col_idx, m in enumerate(header[1:-2], start=1):
                if col_idx < len(row):
                    try:
                        rel_matrix[mid_id][m] = int(row[col_idx].strip())
                    except ValueError:
                        pass
            m_imgs = get_mid_images(fpath, mid_id)
            all_family_images[fid].extend(m_imgs)

        # Generate parent-child kin pairs for this family
        kin_pairs = []
        for m1 in mids:
            for m2 in mids:
                if m1 == m2:
                    continue
                code = rel_matrix.get(m1, {}).get(m2, 0)
                if code in (1, 4):
                    if code == 1:
                        child_id, parent_id = m1, m2
                    else:
                        parent_id, child_id = m1, m2

                    parent_gender = genders.get(parent_id, "m")
                    child_gender = genders.get(child_id, "f")

                    p_imgs = get_mid_images(fpath, parent_id)
                    c_imgs = get_mid_images(fpath, child_id)

                    if parent_gender == "m" and child_gender == "f":
                        rel_str = "fd"
                    elif parent_gender == "m" and child_gender == "m":
                        rel_str = "fs"
                    elif parent_gender == "f" and child_gender == "f":
                        rel_str = "md"
                    else:
                        rel_str = "ms"

                    for p_img in p_imgs:
                        for c_img in c_imgs:
                            kin_pairs.append((p_img, c_img, 1, rel_str))

        family_kin_pairs[fid] = kin_pairs

    # Separate kin pairs into train and test
    train_kin = []
    test_kin = []
    for fid, pairs in family_kin_pairs.items():
        if fid in train_families:
            train_kin.extend(pairs)
        else:
            test_kin.extend(pairs)

    # Generate non-kin pairs for train and test SEPARATELY
    rng_nonkin = random.Random(seed + 1)
    fid_list = [f for f in families if f in all_family_images and all_family_images[f]]
    train_fid_list = [f for f in fid_list if f in train_families]
    test_fid_list = [f for f in fid_list if f in test_families]

    def generate_nonkin(kin_pairs, source_fids, family_imgs, rng_gen):
        """Generate balanced non-kin pairs from a set of families."""
        nonkin = []
        for p1, p2, _, rel_str in kin_pairs:
            p1_fid = os.path.basename(os.path.dirname(os.path.dirname(p1)))
            other_fids = [f for f in source_fids if f != p1_fid and family_imgs.get(f)]
            if not other_fids:
                continue
            other_fid = rng_gen.choice(other_fids)
            other_img = rng_gen.choice(family_imgs[other_fid])
            nonkin.append((p1, other_img, 0, rel_str))
        return nonkin

    train_nonkin = generate_nonkin(train_kin, train_fid_list, all_family_images, rng_nonkin)
    test_nonkin = generate_nonkin(test_kin, test_fid_list, all_family_images, rng_nonkin)

    train_pairs = train_kin + train_nonkin
    test_pairs = test_kin + test_nonkin

    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)

    split_info = {
        "total_families": len(families),
        "train_families": len(train_families),
        "test_families": len(test_families),
        "train_kin": len(train_kin),
        "train_nonkin": len(train_nonkin),
        "train_total": len(train_pairs),
        "test_kin": len(test_kin),
        "test_nonkin": len(test_nonkin),
        "test_total": len(test_pairs),
    }

    print(f"  [FIW-TRAIN] {len(train_pairs)} pairs (kin={len(train_kin)}, non-kin={len(train_nonkin)})")
    print(f"  [FIW-TEST]  {len(test_pairs)} pairs (kin={len(test_kin)}, non-kin={len(test_nonkin)})")

    return train_pairs, test_pairs, split_info


# =============================================================================
# Training Utilities (from train_hybrid_improved.py)
# =============================================================================

def quantum_discrimination_loss(pred_kin, pred_nonkin, margin=0.2):
    """Quantum Discrimination Loss: maximizes separation between kin/non-kin predictions."""
    mean_kin = torch.mean(pred_kin)
    mean_nonkin = torch.mean(pred_nonkin)
    separation = mean_kin - mean_nonkin
    loss = torch.relu(margin - separation)
    return loss


def get_quantum_discrimination_loss(model, emb1, emb2, rels, labels):
    """Computes quantum discrimination loss for a batch."""
    model.eval()
    with torch.no_grad():
        kin_mask = (labels == 1).squeeze()
        nonkin_mask = (labels == 0).squeeze()

        if kin_mask.sum() == 0 or nonkin_mask.sum() == 0:
            return torch.tensor(0.0, device=emb1.device)

        pred_kin = model(emb1[kin_mask], emb2[kin_mask], rels[kin_mask])
        pred_nonkin = model(emb1[nonkin_mask], emb2[nonkin_mask], rels[nonkin_mask])

    return quantum_discrimination_loss(pred_kin, pred_nonkin)


def smooth_labels(labels, epsilon=0.05):
    """Apply label smoothing."""
    return labels * (1.0 - epsilon) + (1.0 - labels) * epsilon


def find_optimal_threshold(y_true, y_scores):
    """Finds optimal threshold using Youden's J-statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_thresh = thresholds[best_idx]
    best_J = J[best_idx]
    pred_labels = (y_scores >= best_thresh).astype(float)
    opt_acc = accuracy_score(y_true, pred_labels) * 100
    return best_thresh, opt_acc, best_J


def compute_per_relation_accuracy(preds, labels, rels, threshold=0.5):
    """Computes accuracy per kinship relation category."""
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


def train_single_fold(
    model, train_emb1, train_emb2, train_y, train_rel,
    val_emb1, val_emb2, val_y, val_rel,
    args, fold_num=0, save_path=None,
):
    """Trains the model for a single fold and returns metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Device] Training Fold {fold_num+1} on: {device}")
    model = model.to(device)

    val_emb1_dev = val_emb1.to(device)
    val_emb2_dev = val_emb2.to(device)
    val_rel_dev = val_rel.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Check if checkpoint already exists for this fold
    if save_path and os.path.exists(save_path) and not getattr(args, "force_retrain", False):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_num+1}: Found existing checkpoint at {os.path.basename(save_path)}. Skipping training.")
        print(f"{'='*60}")
        state_dict = torch.load(save_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        val_emb1_dev = val_emb1.to(device)
        val_emb2_dev = val_emb2.to(device)
        val_rel_dev = val_rel.to(device)

        with torch.no_grad():
            val_preds = model(val_emb1_dev, val_emb2_dev, val_rel_dev).cpu()
            val_preds = torch.clamp(val_preds, 1e-7, 1.0 - 1e-7)

        val_preds_np = val_preds.numpy().flatten()
        val_y_np = val_y.numpy().flatten()

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

        try:
            opt_thresh, opt_acc, opt_J = find_optimal_threshold(val_y_np, val_preds_np)
        except Exception:
            opt_thresh, opt_acc, opt_J = 0.5, acc, 0.0

        per_rel = compute_per_relation_accuracy(val_preds, val_y, val_rel, threshold=0.5)

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
            "best_val_acc": acc,
            "per_relation": per_rel,
            "loss_history": [],
            "acc_history": [],
            "val_acc_history": [],
        }

        print(f"    Fold {fold_num+1} Checkpoint Metrics:")
        print(f"      Accuracy (t=0.5):   {acc:.2f}%")
        print(f"      Accuracy (optimal): {opt_acc:.2f}% (threshold={opt_thresh:.3f})")
        print(f"      Precision: {precision*100:.2f}%")
        print(f"      Recall   : {recall*100:.2f}%")
        print(f"      F1       : {f1*100:.2f}%")
        print(f"      ROC-AUC  : {roc_auc_val:.4f}")
        return fold_results

    train_dataset = TensorDataset(train_emb1, train_emb2, train_y, train_rel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    loss_history = []
    acc_history = []
    val_acc_history = []

    best_val_acc = 0.0
    best_state_dict = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_num+1}: Training ({len(train_y)} train, {len(val_y)} val)")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        correct = 0
        total = 0

        for batch_emb1, batch_emb2, batch_y, batch_rel in train_loader:
            batch_emb1 = batch_emb1.to(device)
            batch_emb2 = batch_emb2.to(device)
            batch_y = batch_y.to(device)
            batch_rel = batch_rel.to(device)

            optimizer.zero_grad()

            # Apply augmentation
            if args.augment:
                noise = torch.randn_like(batch_emb1) * 0.01
                aug_emb1 = batch_emb1 + noise
                aug_emb2 = batch_emb2 + noise
            else:
                aug_emb1, aug_emb2 = batch_emb1, batch_emb2

            # Forward pass
            preds = model(aug_emb1, aug_emb2, batch_rel)
            preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)

            # Apply label smoothing
            if args.label_smoothing > 0:
                targets = smooth_labels(batch_y, epsilon=args.label_smoothing)
            else:
                targets = batch_y

            # Main loss (BCE)
            loss = criterion(preds, targets)

            # Quantum discrimination loss
            if args.quantum_loss_weight > 0:
                q_loss = get_quantum_discrimination_loss(
                    model, aug_emb1, aug_emb2, batch_rel, batch_y
                )
                loss = loss + args.quantum_loss_weight * q_loss

            # Physics-informed regularization
            if args.physics_reg_weight > 0:
                physics_loss = model.physics_regularization()
                loss = loss + args.physics_reg_weight * physics_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pred_labels = (preds >= 0.5).float()
            correct += (pred_labels == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()

        epoch_loss = np.mean(epoch_losses)
        epoch_acc = (correct / total) * 100
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_emb1_dev, val_emb2_dev, val_rel_dev)
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
            print(f"    [Early Stop] No improvement for {args.patience} epochs at epoch {epoch}.")
            break

    # Load best model for evaluation
    if best_state_dict:
        model.load_state_dict(best_state_dict)

    # Final evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(val_emb1_dev, val_emb2_dev, val_rel_dev).cpu()
        val_preds = torch.clamp(val_preds, 1e-7, 1.0 - 1e-7)

    val_preds_np = val_preds.cpu().numpy().flatten()
    val_y_np = val_y.numpy().flatten()

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

    try:
        opt_thresh, opt_acc, opt_J = find_optimal_threshold(val_y_np, val_preds_np)
    except Exception:
        opt_thresh, opt_acc, opt_J = 0.5, acc, 0.0

    per_rel = compute_per_relation_accuracy(val_preds, val_y, val_rel, threshold=0.5)

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
        "loss_history": loss_history,
        "acc_history": acc_history,
        "val_acc_history": val_acc_history,
    }

    print(f"\n    Fold {fold_num+1} Results:")
    print(f"      Accuracy (t=0.5):   {acc:.2f}%")
    print(f"      Accuracy (optimal): {opt_acc:.2f}% (threshold={opt_thresh:.3f})")
    print(f"      Precision: {precision*100:.2f}%")
    print(f"      Recall   : {recall*100:.2f}%")
    print(f"      F1       : {f1*100:.2f}%")
    print(f"      ROC-AUC  : {roc_auc_val:.4f}")
    print(f"      Per-Relation (t=0.5):")
    for rel, stats in per_rel.items():
        print(f"        {rel}: {stats['accuracy']:.1f}% ({stats['count']} pairs)")

    return fold_results


# =============================================================================
# Plotting
# =============================================================================

def plot_roc_curve(labels, scores, save_path, title="ROC Curve -- FIW Retrained Ensemble"):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
            label=f"Ensemble (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(labels, scores, threshold, save_path, title="Confusion Matrix"):
    """Plot confusion matrix."""
    preds = (scores >= threshold).astype(float)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    matrix = np.array([[tn, fp], [fn, tp]])
    total = matrix.sum()
    acc = accuracy_score(labels, preds) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Kin", "Kin"])
    ax.set_yticklabels(["Non-Kin", "Kin"])
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(f"{title}\n(Acc: {acc:.1f}%)", fontweight="bold")

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            pct = val / total * 100
            ax.text(j, i, f"{int(val)}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if val > total / 4 else "black",
                    fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fidelity_distribution(labels, scores, threshold, save_path):
    """Plot fidelity score distributions for kin vs non-kin."""
    fig, ax = plt.subplots(figsize=(9, 5))

    kin_scores = scores[labels == 1] * 100
    nonkin_scores = scores[labels == 0] * 100

    ax.hist(kin_scores, bins=40, alpha=0.65, label=f"Kin (n={len(kin_scores)})",
            color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.hist(nonkin_scores, bins=40, alpha=0.65, label=f"Non-Kin (n={len(nonkin_scores)})",
            color="#F44336", edgecolor="white", linewidth=0.5)

    ax.axvline(threshold * 100, color="black", linestyle="--",
               linewidth=1.5, label=f"Threshold ({threshold*100:.1f}%)")

    gap = np.mean(kin_scores) - np.mean(nonkin_scores)
    ax.set_xlabel("Quantum Fidelity Score (%)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(
        f"Fidelity Distribution -- FIW Unseen Test\n"
        f"Gap: {gap:.1f}pp | Kin: {np.mean(kin_scores):.1f}% | Non-Kin: {np.mean(nonkin_scores):.1f}%",
        fontweight="bold", fontsize=12
    )
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_relation_accuracy(labels, scores, relations, threshold, save_path):
    """Plot per-relation accuracy bar chart."""
    rel_names = ["FD", "FS", "MD", "MS"]
    rel_full = ["Father-Daughter", "Father-Son", "Mother-Daughter", "Mother-Son"]
    preds = (scores >= threshold).astype(float)

    accs = []
    counts = []
    for i in range(4):
        mask = relations == i
        if mask.sum() > 0:
            acc = np.mean(preds[mask] == labels[mask]) * 100
            accs.append(acc)
            counts.append(int(mask.sum()))
        else:
            accs.append(0)
            counts.append(0)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(4), accs, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)

    for bar, acc, count in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%\n(n={count})", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_xticks(range(4))
    ax.set_xticklabels(rel_full, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Per-Relation Accuracy -- FIW Unseen Test", fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(accs) + 10 if accs else 100)
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(50, color="red", linestyle=":", alpha=0.5, label="Random (50%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_dynamics(all_fold_results, save_path):
    """Plot training dynamics across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    # Loss curves
    for i, result in enumerate(all_fold_results):
        axes[0].plot(result["loss_history"], color=colors[i % len(colors)],
                     alpha=0.7, label=f"Fold {i+1}")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Loss", fontweight="bold")
    axes[0].set_title("Training Loss per Fold", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation accuracy curves
    for i, result in enumerate(all_fold_results):
        axes[1].plot(result["val_acc_history"], color=colors[i % len(colors)],
                     alpha=0.7, label=f"Fold {i+1} (best={result['best_val_acc']:.1f}%)")
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Validation Accuracy (%)", fontweight="bold")
    axes[1].set_title("Validation Accuracy per Fold", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training Dynamics -- FIW Retrained Model", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# CLI & Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain Quantum Kinship Ensemble with FIW 50% + KFW-II + TSKinFace"
    )
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-families", type=int, default=200, help="Max TSKinFace families")
    parser.add_argument("--encoding-mode", default="entangled", choices=["entangled", "product"])
    parser.add_argument("--projection", default="quantum_inspired_attention",
                        choices=["quantum_inspired_attention", "simple"])
    parser.add_argument("--cross-val-folds", type=int, default=5)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--quantum-loss-weight", type=float, default=0.3)
    parser.add_argument("--physics-reg-weight", type=float, default=0.1)
    parser.add_argument("--fiw-root", type=str,
                        default=os.path.join(project_root, "public"))
    parser.add_argument("--train-ratio", type=float, default=0.5,
                        help="Fraction of FIW families for training (default: 0.5)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retrain all folds even if fold weight checkpoints exist")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_dir = os.path.join(project_root, "weights")
    results_dir = os.path.join(project_root, "results", "fiw_retrained_metrics")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    KFW2 = os.path.join(project_root, "KinFaceW-II")
    TSKIN = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")

    print("=" * 72)
    print("  QUANTUM KINSHIP -- RETRAIN WITH FIW 50% + EXISTING DATA")
    print("=" * 72)
    print(f"  Configuration:")
    print(f"    Encoding mode    : {args.encoding_mode}")
    print(f"    Projection       : {args.projection}")
    print(f"    Qubits           : {args.n_qubits}")
    print(f"    Epochs           : {args.epochs}")
    print(f"    Batch size       : {args.batch_size}")
    print(f"    Learning rate    : {args.lr}")
    print(f"    Cross-val folds  : {args.cross_val_folds}")
    print(f"    FIW train ratio  : {args.train_ratio}")
    print(f"    Patience         : {args.patience}")
    print("-" * 72)

    # =========================================================================
    # 1. Load Datasets
    # =========================================================================
    print("\n[1/6] Loading datasets...")

    # Existing training data
    existing_train_pairs = []
    if os.path.exists(KFW2):
        kfw2_pairs = load_kinfacew_pairs(KFW2)
        existing_train_pairs.extend(kfw2_pairs)
        print(f"  KinFaceW-II: {len(kfw2_pairs)} pairs")
    else:
        print("  [WARNING] KinFaceW-II not found, skipping")

    if os.path.exists(TSKIN):
        ts_pairs = load_tskinface_pairs(TSKIN, max_families=args.max_families)
        existing_train_pairs.extend(ts_pairs)
        print(f"  TSKinFace:   {len(ts_pairs)} pairs")
    else:
        print("  [WARNING] TSKinFace not found, skipping")

    # FIW split
    print(f"\n  Loading FIW and splitting {int(args.train_ratio*100)}/{int((1-args.train_ratio)*100)}...")
    fiw_train_pairs, fiw_test_pairs, split_info = load_fiw_pairs_split(
        args.fiw_root, train_ratio=args.train_ratio, seed=42
    )

    if len(fiw_train_pairs) == 0 or len(fiw_test_pairs) == 0:
        print("  [ERROR] FIW split failed!")
        sys.exit(1)

    # Combine training data
    all_train_pairs = existing_train_pairs + fiw_train_pairs
    random.Random(42).shuffle(all_train_pairs)

    print(f"\n  {'='*50}")
    print(f"  DATASET SUMMARY")
    print(f"  {'='*50}")
    print(f"  Existing (KFW-II + TSKinFace) : {len(existing_train_pairs)} pairs")
    print(f"  FIW Train ({int(args.train_ratio*100)}% families)     : {len(fiw_train_pairs)} pairs")
    print(f"  TOTAL TRAINING                : {len(all_train_pairs)} pairs")
    print(f"  FIW Test ({int((1-args.train_ratio)*100)}% families)      : {len(fiw_test_pairs)} pairs")
    print(f"  {'='*50}")

    # =========================================================================
    # 2. Extract / Cache Embeddings
    # =========================================================================
    print("\n[2/6] Extracting / caching face embeddings...")

    cache_path = os.path.join(weights_dir, "fiw_retrain_emb_cache.pkl")

    all_pairs_for_cache = all_train_pairs + fiw_test_pairs
    unique_paths = set()
    for p1, p2, _, _ in all_pairs_for_cache:
        unique_paths.add(os.path.normcase(os.path.abspath(p1)))
        unique_paths.add(os.path.normcase(os.path.abspath(p2)))

    # Try loading existing FIW cache to bootstrap
    cache = {}
    for existing_cache_name in ["fiw_retrain_emb_cache.pkl", "fiw_emb_cache.pkl", "embeddings_cache.pkl"]:
        existing_cache_path = find_weight_file(existing_cache_name, project_root)
        if os.path.exists(existing_cache_path):
            try:
                with open(existing_cache_path, "rb") as f:
                    raw = pickle.load(f)
                loaded = {os.path.normcase(os.path.abspath(k)): v for k, v in raw.items()}
                cache.update(loaded)
                print(f"  Loaded {len(loaded)} embeddings from {os.path.basename(existing_cache_path)}")
            except Exception as e:
                print(f"  [Warning] Failed to load {existing_cache_name}: {e}")

    paths_to_extract = [p for p in unique_paths if p not in cache]
    print(f"  Total unique images: {len(unique_paths)}")
    print(f"  Already cached:     {len(unique_paths) - len(paths_to_extract)}")
    print(f"  Need extraction:    {len(paths_to_extract)}")

    if len(paths_to_extract) > 0:
        print(f"  Extracting {len(paths_to_extract)} new embeddings...")
        extractor = FaceFeatureExtractor()
        cache = cache_face_embeddings(all_pairs_for_cache, extractor, cache_path)
    else:
        print("  All embeddings cached! Skipping extraction.")
        # Save merged cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass

    # Prepare tensors
    train_emb1, train_emb2, train_y, train_rel = prepare_pair_tensors(all_train_pairs, cache)
    test_emb1, test_emb2, test_y, test_rel = prepare_pair_tensors(fiw_test_pairs, cache)

    n_kin_tr = int(train_y.sum().item())
    n_kin_te = int(test_y.sum().item())
    print(f"\n  Train tensors: {train_emb1.shape[0]} pairs ({n_kin_tr} kin, {len(train_y)-n_kin_tr} non-kin)")
    print(f"  Test tensors:  {test_emb1.shape[0]} pairs ({n_kin_te} kin, {len(test_y)-n_kin_te} non-kin)")

    # =========================================================================
    # 3. K-Fold Cross-Validation Training
    # =========================================================================
    print(f"\n[3/6] {args.cross_val_folds}-Fold Cross-Validation Training")
    print(f"  Training on {train_emb1.shape[0]} pairs (KFW-II + TSKinFace + FIW-train)")
    print(f"  Validating on held-out folds from training set")
    print("-" * 72)

    n_train = len(train_y)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_train)

    all_fold_results = []
    fold_weight_paths = []

    for fold in range(args.cross_val_folds):
        # Create fold split from the combined training data
        fold_size = n_train // args.cross_val_folds
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < args.cross_val_folds - 1 else n_train
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        fold_train_e1 = train_emb1[train_idx]
        fold_train_e2 = train_emb2[train_idx]
        fold_train_y = train_y[train_idx]
        fold_train_r = train_rel[train_idx]

        fold_val_e1 = train_emb1[val_idx]
        fold_val_e2 = train_emb2[val_idx]
        fold_val_y = train_y[val_idx]
        fold_val_r = train_rel[val_idx]

        # Fresh model for each fold
        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        model = HybridKinshipClassifier(
            n_qubits=args.n_qubits,
            encoding_mode=args.encoding_mode,
            projection_type=args.projection,
        )

        fold_save_path = os.path.join(
            weights_dir, f"fiw_retrained_fold{fold}_{args.encoding_mode}.pt"
        )
        fold_weight_paths.append(fold_save_path)

        fold_result = train_single_fold(
            model,
            fold_train_e1, fold_train_e2, fold_train_y, fold_train_r,
            fold_val_e1, fold_val_e2, fold_val_y, fold_val_r,
            args, fold_num=fold, save_path=fold_save_path,
        )
        all_fold_results.append(fold_result)

    # =========================================================================
    # 4. Cross-Validation Summary
    # =========================================================================
    accs = [r["accuracy"] for r in all_fold_results]
    accs_opt = [r["accuracy_optimal"] for r in all_fold_results]
    aucs_val = [r["roc_auc"] for r in all_fold_results]
    f1s = [r["f1"] for r in all_fold_results]
    thresholds = [r["optimal_threshold"] for r in all_fold_results]

    print(f"\n{'='*72}")
    print(f"  CROSS-VALIDATION SUMMARY ({args.cross_val_folds} folds)")
    print(f"{'='*72}")
    print(f"  Accuracy (t=0.5)  : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"  Accuracy (optimal): {np.mean(accs_opt):.2f}% ± {np.std(accs_opt):.2f}%")
    print(f"  ROC-AUC           : {np.mean(aucs_val):.4f} ± {np.std(aucs_val):.4f}")
    print(f"  F1-Score          : {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
    print(f"  Optimal thresholds: {[f'{t:.3f}' for t in thresholds]}")

    # =========================================================================
    # 5. Build Ensemble & Evaluate on Unseen FIW Test
    # =========================================================================
    print(f"\n[4/6] Building Ensemble from {len(fold_weight_paths)} fold models...")

    sub_models = []
    for path in fold_weight_paths:
        if not os.path.exists(path):
            print(f"  [WARNING] Missing fold weight: {path}")
            continue
        m = HybridKinshipClassifier(
            n_qubits=args.n_qubits,
            encoding_mode=args.encoding_mode,
            projection_type=args.projection,
        )
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        m.load_state_dict(state_dict)
        sub_models.append(m)

    ensemble = EnsembleKinshipClassifier(sub_models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = ensemble.to(device)
    ensemble.eval()
    print(f"  Ensemble created with {len(sub_models)} sub-models (eval device: {device})")

    # Evaluate on held-out FIW test
    print(f"\n[5/6] Evaluating Ensemble on UNSEEN FIW Test ({test_emb1.shape[0]} pairs)...")

    with torch.no_grad():
        ensemble_scores = ensemble(test_emb1.to(device), test_emb2.to(device), test_rel.to(device)).cpu()
        ensemble_scores = torch.clamp(ensemble_scores, 1e-7, 1.0 - 1e-7)

    scores_np = ensemble_scores.numpy().flatten()
    labels_np = test_y.numpy().flatten()
    rels_np = test_rel.numpy().argmax(axis=1)

    # Compute metrics at default threshold (0.5)
    preds_05 = (scores_np >= 0.5).astype(float)
    acc_05 = accuracy_score(labels_np, preds_05) * 100
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(
        labels_np, preds_05, average="binary", zero_division=0
    )

    # Find optimal threshold
    try:
        opt_thresh, opt_acc, opt_J = find_optimal_threshold(labels_np, scores_np)
    except Exception:
        opt_thresh, opt_acc, opt_J = 0.5, acc_05, 0.0

    # ROC AUC
    try:
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        roc_auc_test = auc(fpr, tpr)
    except ValueError:
        roc_auc_test = 0.5

    # Confusion matrix
    preds_opt = (scores_np >= opt_thresh).astype(float)
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_opt).ravel()

    # Fidelity gap
    kin_mean_fid = float(np.mean(scores_np[labels_np == 1]))
    nonkin_mean_fid = float(np.mean(scores_np[labels_np == 0]))

    print(f"\n  {'='*60}")
    print(f"  ENSEMBLE RESULTS ON UNSEEN FIW TEST ({len(labels_np)} pairs)")
    print(f"  {'='*60}")
    print(f"  Accuracy (t=0.5)     : {acc_05:.2f}%")
    print(f"  Accuracy (optimal)   : {opt_acc:.2f}% (threshold={opt_thresh:.3f})")
    print(f"  ROC-AUC              : {roc_auc_test:.4f}")
    print(f"  Precision            : {prec_05*100:.2f}%")
    print(f"  Recall               : {rec_05*100:.2f}%")
    print(f"  F1-Score             : {f1_05*100:.2f}%")
    print(f"  Confusion (optimal)  : TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Kin Mean Fidelity    : {kin_mean_fid:.4f}")
    print(f"  Non-Kin Mean Fidelity: {nonkin_mean_fid:.4f}")
    print(f"  Fidelity Gap         : {(kin_mean_fid - nonkin_mean_fid)*100:.1f}pp")
    print(f"  {'='*60}")

    # =========================================================================
    # 6. Save Ensemble, Plots, and Results
    # =========================================================================
    print(f"\n[6/6] Saving ensemble, plots, and results...")

    # Save ensemble weights
    ensemble_save_path = os.path.join(weights_dir, "ensemble_kinship_fiw.pt")
    torch.save(ensemble.state_dict(), ensemble_save_path)
    print(f"  Saved ensemble: {ensemble_save_path}")

    # Save ensemble metadata
    meta = {
        "n_models": len(sub_models),
        "n_qubits": args.n_qubits,
        "encoding_mode": args.encoding_mode,
        "projection_type": args.projection,
        "optimal_threshold": float(opt_thresh),
        "training_data": "KinFaceW-II + TSKinFace + FIW-train-50%",
        "test_data": "FIW-test-50%",
        "split_info": split_info,
        "source_models": [os.path.basename(p) for p in fold_weight_paths],
    }
    meta_path = os.path.join(weights_dir, "ensemble_fiw_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    # Generate plots
    print("\n  Generating plots...")
    plot_roc_curve(labels_np, scores_np,
                   os.path.join(results_dir, "roc_curve_fiw_retrained.png"))
    plot_confusion_matrix(labels_np, scores_np, opt_thresh,
                          os.path.join(results_dir, "confusion_matrix_fiw_retrained.png"),
                          title="FIW Retrained Ensemble -- Confusion Matrix")
    plot_fidelity_distribution(labels_np, scores_np, opt_thresh,
                               os.path.join(results_dir, "fidelity_distribution_fiw_retrained.png"))
    plot_per_relation_accuracy(labels_np, scores_np, rels_np, opt_thresh,
                               os.path.join(results_dir, "per_relation_accuracy_fiw_retrained.png"))
    plot_training_dynamics(all_fold_results,
                           os.path.join(results_dir, "training_dynamics_fiw_retrained.png"))

    # Save JSON results
    json_results = {
        "ensemble_test_results": {
            "n_pairs": int(len(labels_np)),
            "accuracy_05": float(acc_05),
            "accuracy_optimal": float(opt_acc),
            "optimal_threshold": float(opt_thresh),
            "roc_auc": float(roc_auc_test),
            "precision": float(prec_05 * 100),
            "recall": float(rec_05 * 100),
            "f1": float(f1_05 * 100),
            "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "kin_mean_fidelity": kin_mean_fid,
            "nonkin_mean_fidelity": nonkin_mean_fid,
        },
        "cross_validation": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "accuracy_optimal_mean": float(np.mean(accs_opt)),
            "accuracy_optimal_std": float(np.std(accs_opt)),
            "auc_mean": float(np.mean(aucs_val)),
            "auc_std": float(np.std(aucs_val)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        },
        "dataset_info": {
            "existing_train_pairs": len(existing_train_pairs),
            "fiw_train_pairs": len(fiw_train_pairs),
            "total_train_pairs": len(all_train_pairs),
            "fiw_test_pairs": len(fiw_test_pairs),
            "split_info": split_info,
        },
        "config": {
            "encoding_mode": args.encoding_mode,
            "projection": args.projection,
            "n_qubits": args.n_qubits,
            "folds": args.cross_val_folds,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "label_smoothing": args.label_smoothing,
            "quantum_loss_weight": args.quantum_loss_weight,
            "physics_reg_weight": args.physics_reg_weight,
        },
        "comparison_with_old": {
            "old_model": "ensemble_kinship_full.pt (trained on KFW-II + TSKinFace only)",
            "old_fiw_accuracy": 65.07,
            "old_fiw_auc": 0.7217,
            "new_model": "ensemble_kinship_fiw.pt (trained on KFW-II + TSKinFace + FIW-50%)",
            "new_fiw_accuracy": float(acc_05),
            "new_fiw_auc": float(roc_auc_test),
            "improvement_accuracy": float(acc_05 - 65.07),
            "improvement_auc": float(roc_auc_test - 0.7217),
        },
    }

    json_path = os.path.join(results_dir, "fiw_retrained_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # Final summary
    print(f"\n{'='*72}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*72}")
    print(f"  Old Model (3k train pairs):")
    print(f"    FIW Accuracy: 65.07%  |  AUC: 0.7217")
    print(f"  New Model ({len(all_train_pairs)} train pairs):")
    print(f"    FIW Accuracy: {acc_05:.2f}%  |  AUC: {roc_auc_test:.4f}")
    print(f"    Improvement:  {acc_05-65.07:+.2f}%  |  AUC: {roc_auc_test-0.7217:+.4f}")
    print(f"\n  Ensemble saved to: {ensemble_save_path}")
    print(f"  Results saved to:  {results_dir}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
