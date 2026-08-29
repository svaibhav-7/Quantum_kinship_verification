# -*- coding: utf-8 -*-
"""
=============================================================================
  QUANTUM KINSHIP VERIFICATION -- COMPREHENSIVE MULTI-DATASET EVALUATION
  WITH PUBLICATION-QUALITY METRIC PLOTS
=============================================================================

Evaluates the ensemble model across ALL available datasets (KinFaceW-I,
KinFaceW-II, TSKinFace) from scratch and generates publication-quality plots:
  - ROC Curves (per-dataset + combined)
  - Confusion Matrices
  - Per-Relation Accuracy Bar Charts
  - Fidelity Score Distribution Histograms
  - Precision-Recall Curves
  - Accuracy vs. Threshold Curves

Usage:
  python scripts/evaluate_all_datasets.py
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

# Force UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from src.data_loaders import get_relation_category, load_kinfacew_pairs, load_tskinface_pairs
from src.models_improved import (
    EnsembleKinshipClassifier,
    FaceFeatureExtractor,
    HybridKinshipClassifier,
)


# =============================================================================
# Constants
# =============================================================================

OPTIMAL_THRESHOLD = 0.5279678702354431
RELATION_LABELS = {0: "Father-Daughter", 1: "Father-Son",
                   2: "Mother-Daughter", 3: "Mother-Son"}
RELATION_SHORT = {0: "FD", 1: "FS", 2: "MD", 3: "MS"}

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "KinFaceW-I": "#2196F3",
    "KinFaceW-II": "#FF9800",
    "TSKinFace": "#4CAF50",
    "Combined": "#9C27B0",
    "kin": "#2196F3",
    "nonkin": "#F44336",
}


# =============================================================================
# Model Loading
# =============================================================================

def load_ensemble(weights_dir):
    """Load the single bundled ensemble model."""
    ensemble_path = os.path.join(weights_dir, "ensemble_kinship_full.pt")
    meta_path = os.path.join(weights_dir, "ensemble_metadata.json")

    if os.path.exists(ensemble_path) and os.path.exists(meta_path):
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
        print(f"  Loaded ensemble with {n_models} sub-models")
        return ensemble_model
    else:
        print("  [ERROR] ensemble_kinship_full.pt or ensemble_metadata.json not found!")
        sys.exit(1)


# =============================================================================
# Evaluation Engine
# =============================================================================

def evaluate_dataset(name, pairs, ensemble, extractor):
    """Evaluate ensemble on a list of (img1, img2, label, rel) pairs from scratch."""
    print(f"\n  [{name}] Evaluating {len(pairs)} pairs from scratch...")
    t0 = time.perf_counter()

    all_labels = []
    all_scores = []
    all_relations = []
    skipped = 0
    report_interval = max(1, len(pairs) // 10)

    for idx, (p1, p2, label, rel_str) in enumerate(pairs):
        if (idx + 1) % report_interval == 0 or idx == 0:
            pct = ((idx + 1) / len(pairs)) * 100
            print(f"    Processing {idx+1}/{len(pairs)} ({pct:.0f}%)...")

        try:
            emb1 = extractor.extract(p1)
            emb2 = extractor.extract(p2)
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

    elapsed = time.perf_counter() - t0
    if skipped > 0:
        print(f"    [Info] Skipped {skipped} pairs")
    print(f"    Done in {elapsed:.1f}s ({elapsed/max(1,len(all_labels))*1000:.0f}ms/pair)")

    return np.array(all_labels), np.array(all_scores), np.array(all_relations)


# =============================================================================
# Metric Computation
# =============================================================================

def compute_metrics(labels, scores, threshold=OPTIMAL_THRESHOLD):
    """Compute all classification metrics."""
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
# Plotting Functions
# =============================================================================

def plot_roc_curves(results, save_path):
    """Plot ROC curves for each dataset + combined."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, data in results.items():
        m = data["metrics"]
        ax.plot(m["fpr"], m["tpr"],
                label=f'{name} (AUC = {m["roc_auc"]:.4f}, n={m["n_pairs"]})',
                linewidth=2.2, color=COLORS.get(name, "#666"))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Quantum Kinship Ensemble Model")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(results, save_path):
    """Plot confusion matrices side by side."""
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

    plt.suptitle("Confusion Matrices -- Quantum Kinship Ensemble", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fidelity_distributions(results, save_path):
    """Plot histogram of fidelity scores for Kin vs Non-Kin."""
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
                color=COLORS["kin"], edgecolor="white", linewidth=0.5)
        ax.hist(nonkin_scores, bins=30, alpha=0.65, label=f"Non-Kin (n={len(nonkin_scores)})",
                color=COLORS["nonkin"], edgecolor="white", linewidth=0.5)

        ax.axvline(OPTIMAL_THRESHOLD * 100, color="black", linestyle="--",
                   linewidth=1.5, label=f"Threshold ({OPTIMAL_THRESHOLD*100:.1f}%)")

        gap = np.mean(kin_scores) - np.mean(nonkin_scores)
        ax.set_xlabel("Quantum Fidelity Score (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nGap: {gap:.1f}pp | Kin avg: {np.mean(kin_scores):.1f}% | Non-Kin avg: {np.mean(nonkin_scores):.1f}%")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.suptitle("Fidelity Score Distributions -- Kin vs Non-Kin", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_relation_accuracy(results, save_path):
    """Plot per-relation accuracy bars for each dataset."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    bar_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for ax, (name, data) in zip(axes, results.items()):
        labels = data["labels"]
        scores = data["scores"]
        relations = data["relations"]

        rel_accs = []
        rel_names = []
        for cat_id in range(4):
            mask = relations == cat_id
            if mask.sum() > 0:
                preds = (scores[mask] >= OPTIMAL_THRESHOLD).astype(float)
                acc = accuracy_score(labels[mask], preds) * 100
                rel_accs.append(acc)
                rel_names.append(f"{RELATION_SHORT[cat_id]}\n(n={mask.sum()})")
            else:
                rel_accs.append(0)
                rel_names.append(f"{RELATION_SHORT[cat_id]}\n(n=0)")

        bars = ax.bar(rel_names, rel_accs, color=bar_colors, edgecolor="white", linewidth=1.5)
        for bar, acc in zip(bars, rel_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{name}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(50, color="red", linestyle=":", alpha=0.4, label="Random (50%)")

    plt.suptitle("Per-Relation Accuracy -- Quantum Kinship Ensemble", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_precision_recall_curves(results, save_path):
    """Plot Precision-Recall curves for each dataset."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, data in results.items():
        prec, rec, _ = precision_recall_curve(data["labels"], data["scores"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec,
                label=f'{name} (PR-AUC = {pr_auc:.4f})',
                linewidth=2.2, color=COLORS.get(name, "#666"))

    baseline = sum(results[list(results.keys())[0]]["labels"]) / len(results[list(results.keys())[0]]["labels"])
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.4, label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- Quantum Kinship Ensemble")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_vs_threshold(results, save_path):
    """Plot accuracy as a function of threshold for each dataset."""
    fig, ax = plt.subplots(figsize=(8, 6))
    thresholds = np.linspace(0.01, 0.99, 200)

    for name, data in results.items():
        accs = []
        for t in thresholds:
            preds = (data["scores"] >= t).astype(float)
            accs.append(accuracy_score(data["labels"], preds) * 100)

        ax.plot(thresholds * 100, accs, label=name, linewidth=2, color=COLORS.get(name, "#666"))

    ax.axvline(OPTIMAL_THRESHOLD * 100, color="black", linestyle="--", linewidth=1.5,
               alpha=0.6, label=f"Optimal Threshold ({OPTIMAL_THRESHOLD*100:.1f}%)")
    ax.set_xlabel("Decision Threshold (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Decision Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([40, 85])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary_dashboard(results, save_path):
    """Create a single-page summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # 1. ROC Curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in results.items():
        m = data["metrics"]
        ax1.plot(m["fpr"], m["tpr"], label=f'{name} ({m["roc_auc"]:.3f})',
                 linewidth=2, color=COLORS.get(name, "#666"))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_title("ROC Curves")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # 2. Fidelity Distributions (combined)
    ax2 = fig.add_subplot(gs[0, 1])
    all_labels = np.concatenate([d["labels"] for d in results.values()])
    all_scores = np.concatenate([d["scores"] for d in results.values()])
    ax2.hist(all_scores[all_labels == 1] * 100, bins=40, alpha=0.6,
             color=COLORS["kin"], label="Kin", edgecolor="white")
    ax2.hist(all_scores[all_labels == 0] * 100, bins=40, alpha=0.6,
             color=COLORS["nonkin"], label="Non-Kin", edgecolor="white")
    ax2.axvline(OPTIMAL_THRESHOLD * 100, color="black", linestyle="--", linewidth=1.5)
    ax2.set_title("Fidelity Distribution (All Datasets)")
    ax2.set_xlabel("Quantum Fidelity (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # 3. Metrics table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    table_data = []
    headers = ["Dataset", "Pairs", "Acc%", "AUC", "F1%"]
    for name, data in results.items():
        m = data["metrics"]
        table_data.append([name, str(m["n_pairs"]), f'{m["accuracy"]:.1f}',
                           f'{m["roc_auc"]:.3f}', f'{m["f1"]:.1f}'])
    table = ax3.table(cellText=table_data, colLabels=headers, loc="center",
                      cellLoc="center", colColours=["#E3F2FD"] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax3.set_title("Performance Summary", pad=20)

    # 4. Per-relation accuracy (combined)
    ax4 = fig.add_subplot(gs[1, 0])
    all_relations = np.concatenate([d["relations"] for d in results.values()])
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    rel_accs = []
    rel_names = []
    for cat_id in range(4):
        mask = all_relations == cat_id
        if mask.sum() > 0:
            preds = (all_scores[mask] >= OPTIMAL_THRESHOLD).astype(float)
            acc = accuracy_score(all_labels[mask], preds) * 100
            rel_accs.append(acc)
            rel_names.append(f"{RELATION_SHORT[cat_id]}\n(n={mask.sum()})")
    bars = ax4.bar(rel_names, rel_accs, color=bar_colors[:len(rel_accs)])
    for bar, acc in zip(bars, rel_accs):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax4.set_ylim(0, 100)
    ax4.set_title("Per-Relation Accuracy (Combined)")
    ax4.axhline(50, color="red", linestyle=":", alpha=0.3)
    ax4.grid(True, axis="y", alpha=0.2)

    # 5. Confusion matrix (combined)
    ax5 = fig.add_subplot(gs[1, 1])
    preds_all = (all_scores >= OPTIMAL_THRESHOLD).astype(float)
    cm = confusion_matrix(all_labels, preds_all)
    im = ax5.imshow(cm, cmap="Blues")
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["Non-Kin", "Kin"])
    ax5.set_yticklabels(["Non-Kin", "Kin"])
    ax5.set_xlabel("Predicted")
    ax5.set_ylabel("Actual")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)",
                     ha="center", va="center", fontsize=11,
                     color="white" if cm[i, j] > total / 4 else "black", fontweight="bold")
    ax5.set_title(f"Confusion Matrix (All {total} pairs)")

    # 6. Accuracy vs Threshold
    ax6 = fig.add_subplot(gs[1, 2])
    thresholds = np.linspace(0.01, 0.99, 200)
    for name, data in results.items():
        accs = [accuracy_score(data["labels"], (data["scores"] >= t).astype(float)) * 100
                for t in thresholds]
        ax6.plot(thresholds * 100, accs, label=name, linewidth=1.5, color=COLORS.get(name, "#666"))
    ax6.axvline(OPTIMAL_THRESHOLD * 100, color="black", linestyle="--", alpha=0.5)
    ax6.set_title("Accuracy vs. Threshold")
    ax6.set_xlabel("Threshold (%)")
    ax6.set_ylabel("Accuracy (%)")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.2)

    plt.suptitle("Quantum Kinship Verification -- Ensemble Model Evaluation Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    results_dir = os.path.join(project_root, "results", "evaluation_plots")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("  QUANTUM KINSHIP -- COMPREHENSIVE MULTI-DATASET EVALUATION")
    print("  (All embeddings from scratch | Publication-quality plots)")
    print("=" * 72)

    # 1. Init
    print("\n[INIT] Loading FaceNet Feature Extractor...")
    extractor = FaceFeatureExtractor()

    print("\n[INIT] Loading Ensemble Model...")
    ensemble = load_ensemble(weights_dir)

    # 2. Load ALL datasets
    print("\n[DATA] Loading datasets...")
    datasets = {}

    # KinFaceW-I (unseen test set)
    kfw1_dir = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    kfw1_pairs = load_kinfacew_pairs(kfw1_dir)
    if kfw1_pairs:
        datasets["KinFaceW-I"] = kfw1_pairs
        print(f"  KinFaceW-I:  {len(kfw1_pairs)} pairs (UNSEEN TEST SET)")

    # KinFaceW-II
    kfw2_dir = os.path.join(project_root, "KinFaceW-II")
    kfw2_pairs = load_kinfacew_pairs(kfw2_dir)
    if kfw2_pairs:
        datasets["KinFaceW-II"] = kfw2_pairs
        print(f"  KinFaceW-II: {len(kfw2_pairs)} pairs")

    # TSKinFace
    tsk_dir = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
    tsk_pairs = load_tskinface_pairs(tsk_dir)
    if tsk_pairs:
        datasets["TSKinFace"] = tsk_pairs
        print(f"  TSKinFace:   {len(tsk_pairs)} pairs")

    total = sum(len(v) for v in datasets.values())
    print(f"\n  TOTAL: {total} pairs across {len(datasets)} datasets")

    # 3. Evaluate each dataset
    print("\n" + "=" * 72)
    print("  EVALUATING ALL DATASETS (FROM SCRATCH)")
    print("=" * 72)

    all_results = {}
    for name, pairs in datasets.items():
        labels, scores, relations = evaluate_dataset(name, pairs, ensemble, extractor)
        metrics = compute_metrics(labels, scores)

        all_results[name] = {
            "labels": labels,
            "scores": scores,
            "relations": relations,
            "metrics": metrics,
        }

        print(f"\n  [{name}] Results:")
        print(f"    Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    Precision: {metrics['precision']:.2f}%")
        print(f"    Recall:    {metrics['recall']:.2f}%")
        print(f"    F1-Score:  {metrics['f1']:.2f}%")

    # 4. Combined metrics
    combined_labels = np.concatenate([d["labels"] for d in all_results.values()])
    combined_scores = np.concatenate([d["scores"] for d in all_results.values()])
    combined_relations = np.concatenate([d["relations"] for d in all_results.values()])
    combined_metrics = compute_metrics(combined_labels, combined_scores)

    print(f"\n  [COMBINED] Results across all {total} pairs:")
    print(f"    Accuracy:  {combined_metrics['accuracy']:.2f}%")
    print(f"    ROC-AUC:   {combined_metrics['roc_auc']:.4f}")
    print(f"    F1-Score:  {combined_metrics['f1']:.2f}%")

    # 5. Generate plots
    print("\n" + "=" * 72)
    print("  GENERATING PUBLICATION-QUALITY PLOTS")
    print("=" * 72)

    plot_roc_curves(all_results, os.path.join(results_dir, "roc_curves.png"))
    plot_confusion_matrices(all_results, os.path.join(results_dir, "confusion_matrices.png"))
    plot_fidelity_distributions(all_results, os.path.join(results_dir, "fidelity_distributions.png"))
    plot_per_relation_accuracy(all_results, os.path.join(results_dir, "per_relation_accuracy.png"))
    plot_precision_recall_curves(all_results, os.path.join(results_dir, "precision_recall_curves.png"))
    plot_accuracy_vs_threshold(all_results, os.path.join(results_dir, "accuracy_vs_threshold.png"))
    plot_summary_dashboard(all_results, os.path.join(results_dir, "summary_dashboard.png"))

    # 6. Save JSON results
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
    json_results["combined"] = {
        "n_pairs": int(len(combined_labels)),
        "accuracy": combined_metrics["accuracy"],
        "roc_auc": combined_metrics["roc_auc"],
        "precision": combined_metrics["precision"],
        "recall": combined_metrics["recall"],
        "f1": combined_metrics["f1"],
        "confusion": combined_metrics["confusion"],
    }

    json_path = os.path.join(results_dir, "multi_dataset_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved JSON results: {json_path}")

    print("\n" + "=" * 72)
    print("  ALL EVALUATIONS AND PLOTS COMPLETE!")
    print(f"  Plots saved to: {results_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
