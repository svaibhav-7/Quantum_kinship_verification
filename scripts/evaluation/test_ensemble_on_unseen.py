# -*- coding: utf-8 -*-
"""
=============================================================================
  QUANTUM KINSHIP VERIFICATION -- UNSEEN DATASET EVALUATION (FIW & KINFACEW-I)
=============================================================================

Evaluates the Quantum Kinship Ensemble model on truly unseen datasets:
1. FIW (Family In the Wild) dataset located in `public/FIDs` (36,900+ Kin pairs + Non-Kin pairs)
2. KinFaceW-I dataset (1,066 pairs)

Usage:
  python scripts/test_ensemble_on_unseen.py --max-fiw-pairs 0  # evaluates ALL 36,000+ FIW pairs
"""

import argparse
import csv
import glob
import json
import os
import pickle
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

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
        os.path.join(project_root, "weights", "active_ensemble"),
        os.path.join(project_root, "weights", "folds"),
        os.path.join(project_root, "weights", "caches"),
        os.path.join(project_root, "weights"),
        os.path.join(project_root, "weights", "archive"),
    ]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(project_root, "weights", filename)

from src.data_loaders import get_relation_category, load_kinfacew_pairs
from src.models_improved import EnsembleKinshipClassifier, FaceFeatureExtractor, HybridKinshipClassifier


# =============================================================================
# Data Loaders
# =============================================================================

def load_fiw_pairs(fiw_root, max_pairs=None):
    """
    Load FIW (Family In the Wild) dataset pairs from `public/FIDs`.

    Parses each family's `mid.csv` to map parent-child relationships (FD, FS, MD, MS).
    Creates a 1:1 balanced set of Kin and Non-Kin pairs across families.
    """
    fids_dir = os.path.join(fiw_root, "FIDs") if os.path.exists(os.path.join(fiw_root, "FIDs")) else fiw_root
    if not os.path.exists(fids_dir):
        print(f"  [WARNING] FIDs directory not found at {fids_dir}")
        return []

    families = sorted([d for d in os.listdir(fids_dir) if os.path.isdir(os.path.join(fids_dir, d))])
    kin_pairs = []
    all_family_images = {}

    for fid in families:
        fpath = os.path.join(fids_dir, fid)
        mid_csv = os.path.join(fpath, "mid.csv")
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
            m_imgs = glob.glob(os.path.join(fpath, f"MID{mid_id}", "*.jpg")) + glob.glob(os.path.join(fpath, f"MID{mid_id}", "*.png"))
            all_family_images[fid].extend(m_imgs)

        # Generate parent-child kin pairs
        for m1 in mids:
            for m2 in mids:
                if m1 == m2:
                    continue
                code = rel_matrix.get(m1, {}).get(m2, 0)
                if code in (1, 4):  # 1 = child of, 4 = parent of
                    if code == 1:
                        child_id, parent_id = m1, m2
                    else:
                        parent_id, child_id = m1, m1

                    parent_gender = genders.get(parent_id, "m")
                    child_gender = genders.get(child_id, "f")

                    p_imgs = glob.glob(os.path.join(fpath, f"MID{parent_id}", "*.jpg"))
                    c_imgs = glob.glob(os.path.join(fpath, f"MID{child_id}", "*.jpg"))

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

    # Generate equal number of Non-Kin Pairs from different families
    nonkin_pairs = []
    fid_list = [f for f in families if f in all_family_images and all_family_images[f]]
    random.seed(42)

    for p1, p2, _, rel_str in kin_pairs:
        p1_fid = os.path.basename(os.path.dirname(os.path.dirname(p1)))
        other_fids = [f for f in fid_list if f != p1_fid]
        if not other_fids:
            continue
        other_fid = random.choice(other_fids)
        other_img = random.choice(all_family_images[other_fid])
        nonkin_pairs.append((p1, other_img, 0, rel_str))

    total_pairs = kin_pairs + nonkin_pairs
    random.seed(42)
    random.shuffle(total_pairs)

    if max_pairs and max_pairs > 0 and len(total_pairs) > max_pairs:
        half = max_pairs // 2
        sampled_kin = [p for p in total_pairs if p[2] == 1][:half]
        sampled_nonkin = [p for p in total_pairs if p[2] == 0][:half]
        total_pairs = sampled_kin + sampled_nonkin
        random.shuffle(total_pairs)

    print(f"  [FIW] Loaded {len(total_pairs)} pairs (kin={sum(1 for p in total_pairs if p[2]==1)}, non-kin={sum(1 for p in total_pairs if p[2]==0)})")
    return total_pairs


# =============================================================================
# Evaluation Engine with Caching
# =============================================================================

def evaluate_dataset(name, pairs, ensemble, extractor, cache_dir=None):
    """Evaluate ensemble on pairs with optional disk embedding caching."""
    print(f"\n  [{name}] Evaluating {len(pairs)} pairs...")
    t0 = time.perf_counter()

    cache_path = os.path.join(cache_dir, f"{name.lower()}_emb_cache.pkl") if cache_dir else None
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"    Loaded {len(cache)} cached face embeddings from: {os.path.basename(cache_path)}")
        except Exception:
            cache = {}

    all_labels = []
    all_scores = []
    all_relations = []
    skipped = 0
    new_extractions = 0
    report_interval = max(1, len(pairs) // 20)

    for idx, (p1, p2, label, rel_str) in enumerate(pairs):
        if (idx + 1) % report_interval == 0 or idx == 0:
            pct = ((idx + 1) / len(pairs)) * 100
            print(f"    Processing {idx+1}/{len(pairs)} ({pct:.0f}%)... [Cached: {len(cache)}]")

        try:
            p1_abs = os.path.normcase(os.path.abspath(p1))
            p2_abs = os.path.normcase(os.path.abspath(p2))

            if p1_abs in cache:
                emb1 = cache[p1_abs]
            else:
                emb1 = extractor.extract(p1)
                cache[p1_abs] = emb1
                new_extractions += 1

            if p2_abs in cache:
                emb2 = cache[p2_abs]
            else:
                emb2 = extractor.extract(p2)
                cache[p2_abs] = emb2
                new_extractions += 1

        except Exception:
            skipped += 1
            continue

        emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
        emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)

        cat = get_relation_category(rel_str, p1)
        rel_vec = [0.0] * 4
        rel_vec[cat] = 1.0
        rel_t = torch.tensor([rel_vec], dtype=torch.float32)

        with torch.no_grad():
            score = ensemble(emb1_t, emb2_t, rel_t).item()

        all_labels.append(label)
        all_scores.append(score)
        all_relations.append(cat)

        # Periodically save cache if new extractions were performed
        if cache_path and new_extractions > 0 and (new_extractions % 500 == 0):
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
            except Exception:
                pass

    # Save final cache
    if cache_path and new_extractions > 0:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
            print(f"    Saved {len(cache)} embeddings to cache")
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    if skipped > 0:
        print(f"    [Info] Skipped {skipped} pairs")
    print(f"    Done in {elapsed:.1f}s ({elapsed/max(1,len(all_labels))*1000:.0f}ms/pair)")

    return np.array(all_labels), np.array(all_scores), np.array(all_relations)


def compute_metrics(labels, scores, threshold=0.5279678702354431):
    """Compute classification metrics."""
    preds = (scores >= threshold).astype(float)
    acc = accuracy_score(labels, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc_val = auc(fpr, tpr)
    except ValueError:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        roc_auc_val = 0.5

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": acc,
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "n_pairs": len(labels),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_roc_curves(results, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for idx, (name, data) in enumerate(results.items()):
        m = data["metrics"]
        c = colors[idx % len(colors)]
        ax.plot(m["fpr"], m["tpr"],
                label=f'{name} (AUC = {m["roc_auc"]:.4f}, n={m["n_pairs"]})',
                linewidth=2.2, color=c)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Unseen Dataset Evaluation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        cm = data["metrics"]["confusion"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        total = matrix.sum()

        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Kin", "Kin"])
        ax.set_yticklabels(["Non-Kin", "Kin"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\n(Acc: {data['metrics']['accuracy']:.1f}%)")

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                pct = val / total * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=12,
                        color="white" if val > total / 4 else "black",
                        fontweight="bold")

    plt.suptitle("Confusion Matrices -- Unseen Datasets", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fidelity_distributions(results, save_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        labels = data["labels"]
        scores = data["scores"]

        kin_scores = scores[labels == 1] * 100
        nonkin_scores = scores[labels == 0] * 100

        ax.hist(kin_scores, bins=30, alpha=0.65, label=f"Kin (n={len(kin_scores)})",
                color="#2196F3", edgecolor="white", linewidth=0.5)
        ax.hist(nonkin_scores, bins=30, alpha=0.65, label=f"Non-Kin (n={len(nonkin_scores)})",
                color="#F44336", edgecolor="white", linewidth=0.5)

        ax.axvline(0.5279678702354431 * 100, color="black", linestyle="--",
                   linewidth=1.5, label=f"Threshold (52.8%)")

        gap = np.mean(kin_scores) - np.mean(nonkin_scores)
        ax.set_xlabel("Quantum Fidelity Score (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nGap: {gap:.1f}pp | Kin: {np.mean(kin_scores):.1f}% | Non-Kin: {np.mean(nonkin_scores):.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.suptitle("Fidelity Score Distributions -- Kin vs Non-Kin", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# CLI & Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test Quantum Kinship Ensemble Model on Unseen Datasets (FIW & KinFaceW-I)"
    )
    parser.add_argument(
        "--kfw1-root",
        type=str,
        default=os.path.join(project_root, "KinFaceW-I", "KinFaceW-I"),
        help="Path to KinFaceW-I dataset root",
    )
    parser.add_argument(
        "--fiw-root",
        type=str,
        default=os.path.join(project_root, "public"),
        help="Path to FIW dataset root (contains public/FIDs)",
    )
    parser.add_argument(
        "--max-fiw-pairs",
        type=int,
        default=0,
        help="Maximum FIW pairs to evaluate (0 for ALL pairs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(project_root, "results", "unseen_metrics"),
        help="Directory to save evaluation metrics and plots",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights_dir = os.path.join(project_root, "weights")

    print("=" * 72)
    print("  QUANTUM KINSHIP -- UNSEEN DATASET EVALUATION (FIW & KINFACEW-I)")
    print("=" * 72)

    # 1. Init Extractor
    print("\n[INIT] Loading FaceNet Feature Extractor...")
    extractor = FaceFeatureExtractor()

    # 2. Load Ensemble
    print("\n[INIT] Loading Ensemble Model...")
    ensemble_path = find_weight_file("ensemble_kinship_full.pt", project_root)
    meta_path = find_weight_file("ensemble_metadata.json", project_root)

    if not os.path.exists(ensemble_path) or not os.path.exists(meta_path):
        print(f"  [ERROR] Ensemble model file not found at: {ensemble_path}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    n_models = meta["n_models"]
    sub_models = [
        HybridKinshipClassifier(
            n_qubits=meta.get("n_qubits", 8),
            encoding_mode=meta.get("encoding_mode", "entangled"),
            projection_type=meta.get("projection_type", "quantum_inspired_attention"),
        )
        for _ in range(n_models)
    ]
    ensemble_model = EnsembleKinshipClassifier(sub_models)
    state_dict = torch.load(ensemble_path, map_location="cpu", weights_only=True)
    ensemble_model.load_state_dict(state_dict)
    ensemble_model.eval()
    print(f"  Loaded single-file ensemble ({n_models} sub-models)")

    # 3. Load Datasets
    print("\n[DATA] Loading unseen datasets...")
    datasets = {}

    # KinFaceW-I (1,066 pairs)
    if os.path.exists(args.kfw1_root):
        kfw1_pairs = load_kinfacew_pairs(args.kfw1_root)
        if kfw1_pairs:
            datasets["KinFaceW-I"] = kfw1_pairs
            print(f"  KinFaceW-I: {len(kfw1_pairs)} pairs")

    # FIW (Family In the Wild)
    fiw_pairs = load_fiw_pairs(args.fiw_root, max_pairs=args.max_fiw_pairs)
    if fiw_pairs:
        datasets["FIW"] = fiw_pairs

    total_pairs = sum(len(v) for v in datasets.values())
    print(f"\n  TOTAL: {total_pairs} pairs across {len(datasets)} dataset(s)")

    if total_pairs == 0:
        print("  [ERROR] No pairs loaded!")
        sys.exit(1)

    # 4. Evaluate each dataset
    print("\n" + "=" * 72)
    print("  EVALUATING UNSEEN DATASETS (WITH EMBEDDING CACHING)")
    print("=" * 72)

    all_results = {}
    for name, pairs in datasets.items():
        labels, scores, relations = evaluate_dataset(name, pairs, ensemble_model, extractor, cache_dir=weights_dir)
        metrics = compute_metrics(labels, scores)

        all_results[name] = {
            "labels": labels,
            "scores": scores,
            "relations": relations,
            "metrics": metrics,
        }

        print(f"\n  [{name}] Results:")
        print(f"    Pairs:     {metrics['n_pairs']}")
        print(f"    Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    Precision: {metrics['precision']:.2f}%")
        print(f"    Recall:    {metrics['recall']:.2f}%")
        print(f"    F1-Score:  {metrics['f1']:.2f}%")

    # 5. Generate plots
    print("\n" + "=" * 72)
    print("  GENERATING PLOTS")
    print("=" * 72)

    plot_roc_curves(all_results, os.path.join(args.output_dir, "roc_curves_unseen.png"))
    plot_confusion_matrices(all_results, os.path.join(args.output_dir, "confusion_matrices_unseen.png"))
    plot_fidelity_distributions(all_results, os.path.join(args.output_dir, "fidelity_distributions_unseen.png"))

    # 6. Save JSON
    json_results = {}
    for name, data in all_results.items():
        m = data["metrics"]
        json_results[name] = {
            "n_pairs": m["n_pairs"],
            "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "confusion": m["confusion"],
            "kin_mean_fidelity": float(np.mean(data["scores"][data["labels"] == 1])),
            "nonkin_mean_fidelity": float(np.mean(data["scores"][data["labels"] == 0])),
        }

    json_path = os.path.join(args.output_dir, "unseen_evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved JSON results: {json_path}")

    print("\n" + "=" * 72)
    print("  EVALUATION COMPLETE!")
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()