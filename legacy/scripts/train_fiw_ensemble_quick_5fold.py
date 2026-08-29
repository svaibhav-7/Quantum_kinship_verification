# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- TRAIN PURE FIW ENSEMBLE (QUICK 5-FOLD)
=================================================================================

Creates a 5-fold ensemble trained exclusively on FIW data for use in the meta-ensemble.
Uses reduced epochs for quick training.

Usage:
  python scripts/training/train_fiw_ensemble_quick_5fold.py
"""

import os
import sys
import json
import random
import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader

# Project root setup (supports .py, Jupyter, Google Colab, and Kaggle)
def setup_project_environment():
    try:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start_dir = os.getcwd()

    # 1. Walk upward to find directory containing src/models_improved.py
    curr = os.path.abspath(start_dir)
    while curr and curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "src", "models_improved.py")):
            if curr not in sys.path:
                sys.path.insert(0, curr)
            try:
                os.chdir(curr)
            except Exception:
                pass
            return curr
        curr = os.path.dirname(curr)

    # 2. Check immediate child directories (e.g. inside Google Colab /content/)
    try:
        for item in os.listdir(start_dir):
            full_path = os.path.join(start_dir, item)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "src", "models_improved.py")):
                if full_path not in sys.path:
                    sys.path.insert(0, full_path)
                try:
                    os.chdir(full_path)
                except Exception:
                    pass
                return full_path
    except Exception:
        pass

    if start_dir not in sys.path:
        sys.path.insert(0, start_dir)
    return start_dir

project_root = setup_project_environment()

from src.models_improved import (
    FaceFeatureExtractor,
    HybridKinshipClassifier,
    EnsembleKinshipClassifier,
)
from src.data_loaders import (
    load_fiw_pairs,
    cache_face_embeddings,
    prepare_pair_tensors,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)


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


def smooth_labels(labels, epsilon=0.05):
    """Apply label smoothing."""
    return labels * (1.0 - epsilon) + (1.0 - labels) * epsilon


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


def train_single_fold(
    model, train_emb1, train_emb2, train_y, train_rel,
    val_emb1, val_emb2, val_y, val_rel,
    args, fold_num=0, save_path=None,
):
    """Trains the model for a single fold and returns metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(train_emb1, train_emb2, train_y, train_rel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

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
            loss = torch.nn.functional.binary_cross_entropy(preds, targets)

            # Quantum Discrimination Loss
            if args.quantum_loss_weight > 0:
                q_loss = get_quantum_discrimination_loss(model, batch_emb1, batch_emb2, batch_rel, batch_y)
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
            val_preds = model(val_emb1.to(device), val_emb2.to(device), val_rel.to(device))
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
                f"  Auto-stopping at epoch {epoch}: validation has not improved for {args.patience} checks"
            )
            break

    # Final evaluation on validation set
    model.eval()
    with torch.no_grad():
        val_preds = model(val_emb1.to(device), val_emb2.to(device), val_rel.to(device))
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
    return fold_results


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


def main():
    """Main function to train FIW-only ensemble."""
    print("=" * 72)
    print("  QUANTUM KINSHIP -- TRAIN PURE FIW ENSEMBLE (QUICK 5-FOLD)")
    print("=" * 72)

    # Configuration - quick training settings
    args = type('Args', (), {})()
    args.n_qubits = 8
    args.epochs = 2          # Reduced for quick training
    args.batch_size = 32     # Reduced for quick training
    args.lr = 2e-4
    args.encoding_mode = "entangled"
    args.projection = "quantum_inspired_attention"
    args.cross_val_folds = 5 # Keep 5-fold
    args.augment = True
    args.label_smoothing = 0.05
    args.quantum_loss_weight = 0.3
    args.physics_reg_weight = 0.1
    args.fiw_root = os.path.join(project_root, "public")
    args.seed = 42
    args.patience = 2        # Reduced for quick training

    print(f"  Configuration:")
    print(f"    Encoding mode    : {args.encoding_mode}")
    print(f"    Projection       : {args.projection}")
    print(f"    Qubits           : {args.n_qubits}")
    print(f"    Epochs           : {args.epochs}")
    print(f"    Batch size       : {args.batch_size}")
    print(f"    Learning rate    : {args.lr}")
    print(f"    Cross-val folds  : {args.cross_val_folds}")
    print(f"    FIW root         : {args.fiw_root}")
    print(f"    Seed             : {args.seed}")
    print("-" * 72)

    weights_dir = os.path.join(project_root, "weights")
    results_dir = os.path.join(project_root, "results", "fiw_ensemble_pure_quick")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # =========================================================================
    # 1. Load FIW Dataset (100% for training)
    # =========================================================================
    print("\n[1/4] Loading FIW dataset...")
    fiw_pairs = load_fiw_pairs(args.fiw_root)  # We don't need the test split for ensemble training
    print(f"  FIW pairs: {len(fiw_pairs)}")

    if len(fiw_pairs) == 0:
        print("  [ERROR] No FIW pairs found!")
        sys.exit(1)

    # =========================================================================
    # 2. Extract / Cache Embeddings
    # =========================================================================
    print("\n[2/4] Extracting / caching face embeddings...")
    cache_path = os.path.join(weights_dir, "fiw_ensemble_emb_cache.pkl")

    # Try loading existing FIW cache to bootstrap
    cache = {}
    for existing_cache_name in ["fiw_emb_cache.pkl", "fiw_retrain_emb_cache.pkl"]:
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

    if len(cache) == 0:
        print("  No existing cache found, extracting all embeddings...")
        extractor = FaceFeatureExtractor()
        cache = cache_face_embeddings(fiw_pairs, extractor, cache_path)
    else:
        # Extract any missing embeddings
        all_paths = set()
        for p1, p2, _, _ in fiw_pairs:
            all_paths.add(os.path.normcase(os.path.abspath(p1)))
            all_paths.add(os.path.normcase(os.path.abspath(p2)))
        paths_to_extract = [p for p in all_paths if p not in cache]
        if paths_to_extract:
            print(f"  Extracting {len(paths_to_extract)} missing embeddings...")
            extractor = FaceFeatureExtractor()
            new_cache = cache_face_embeddings(fiw_pairs, extractor, cache_path)
            cache.update(new_cache)
            # Save updated cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
            except Exception:
                pass
        else:
            print("  All embeddings cached! Skipping extraction.")

    # Prepare tensors
    train_emb1, train_emb2, train_y, train_rel = prepare_pair_tensors(fiw_pairs, cache)

    n_kin = int(train_y.sum().item())
    print(f"\n  Tensors: {train_emb1.shape[0]} pairs ({n_kin} kin, {len(train_y)-n_kin} non-kin)")

    # =========================================================================
    # 3. 5-Fold Cross-Validation Training on FIW Data
    # =========================================================================
    print(f"\n[3/4] {args.cross_val_folds}-Fold Cross-Validation Training on FIW Data")
    print(f"  Training on 100% of FIW data using {args.cross_val_folds}-fold CV")
    print("-" * 72)

    n_train = len(train_y)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n_train)

    all_fold_results = []
    fold_weight_paths = []

    for fold in range(args.cross_val_folds):
        # Create fold split
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

        # Fresh model for each fold with deterministic initialization
        torch.manual_seed(args.seed + fold)
        np.random.seed(args.seed + fold)
        model = HybridKinshipClassifier(
            n_qubits=args.n_qubits,
            encoding_mode=args.encoding_mode,
            projection_type=args.projection,
        )

        # Save each fold's weights
        fold_save_path = os.path.join(
            weights_dir, "folds", f"fiw_ensemble_fold{fold}_{args.encoding_mode}.pt"
        )
        os.makedirs(os.path.dirname(fold_save_path), exist_ok=True)
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

    # =========================================================================
    # 4. Create and Save Ensemble
    # =========================================================================
    print(f"\n[4/4] Creating ensemble from {len(fold_weight_paths)} trained folds...")

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

    # Save ensemble weights
    ensemble_save_path = os.path.join(weights_dir, "active_ensemble", "ensemble_kinship_fiw.pt")
    os.makedirs(os.path.dirname(ensemble_save_path), exist_ok=True)
    torch.save(ensemble.state_dict(), ensemble_save_path)
    print(f"  Saved ensemble: {ensemble_save_path}")

    # Calculate average metrics across folds
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
    print(f"  ROC-AUC          : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  F1-Score         : {np.mean(f1s):.2f}% +/- {np.std(f1s):.2f}%")
    print(f"  Optimal Threshold: {np.mean(thresholds):.4f} +/- {np.std(thresholds):.4f}")

    # Save ensemble metadata
    meta_path = os.path.join(weights_dir, "active_ensemble", "ensemble_fiw_metadata.json")
    meta = {
        "n_models": len(sub_models),
        "n_qubits": args.n_qubits,
        "encoding_mode": args.encoding_mode,
        "projection_type": args.projection,
        "optimal_threshold": float(np.mean(thresholds)),
        "training_data": "100% FIW data (5-fold CV)",
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs)),
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "source_models": [os.path.basename(p) for p in fold_weight_paths],
        "creation_date": str(np.datetime64('now')),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    print("\n[SUCCESS] Pure FIW ensemble training completed!")


if __name__ == "__main__":
    main()