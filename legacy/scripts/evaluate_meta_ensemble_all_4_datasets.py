# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- META-ENSEMBLE EVALUATION ACROSS 4 DATASETS
=================================================================================

Evaluates the Meta-Ensemble Classifier (meta_ensemble_kinship.pt) across 4 major datasets:
  1. KinFaceW-I Benchmark (1,066 pairs)
  2. KinFaceW-II Benchmark (2,000 pairs)
  3. TSKinFace Dataset (1,200 pairs)
  4. Families In the Wild (FIW) Dataset (500 pairs)

Generates:
  - JSON Metrics: results/meta_ensemble_4_datasets_comprehensive_metrics.json
  - Plots:
      1. results/plots/meta_ensemble_4_datasets_roc_curves.png
      2. results/plots/meta_ensemble_4_datasets_metrics_bar_chart.png
      3. results/plots/meta_ensemble_4_datasets_relation_breakdown.png
      4. results/plots/meta_ensemble_4_datasets_confusion_matrices.png

Usage:
  python scripts/evaluation/evaluate_meta_ensemble_all_4_datasets.py
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

# Project Root Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "evaluation", "training", "inference"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import (
    load_kinfacew_pairs,
    load_tskinface_pairs,
    prepare_pair_tensors
)
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight"
})

DATASET_COLORS = {
    "KinFaceW-I Benchmark": "#1976D2",   # Deep Blue
    "KinFaceW-II Benchmark": "#009688",  # Teal
    "TSKinFace Dataset": "#FF9800",      # Orange
    "Families In the Wild (FIW)": "#9C27B0" # Purple
}

def load_meta_ensemble(project_root):
    path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")

    meta_model = MetaEnsembleKinshipClassifier(m1, m2, m3)  # Use default weights, will be overridden by state dict
    state = torch.load(path, map_location="cpu")
    meta_model.load_state_dict(state)
    meta_model.eval()
    return meta_model

def predict_batched(model, emb1, emb2, rels, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    n_samples = len(emb1)
    preds = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_e1 = emb1[i:i+batch_size].to(device)
            b_e2 = emb2[i:i+batch_size].to(device)
            b_rel = rels[i:i+batch_size].to(device)
            p = model(b_e1, b_e2, b_rel)
            preds.append(p.cpu().view(-1))
            
    return torch.cat(preds, dim=0).numpy()

def evaluate_dataset(y_true, preds, rel_tensors):
    fpr, tpr, thresholds = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_thresh = float(thresholds[opt_idx])

    binary_opt = (preds >= opt_thresh).astype(int)
    binary_05 = (preds >= 0.50).astype(int)

    acc_opt = accuracy_score(y_true, binary_opt)
    prec_opt, rec_opt, f1_opt, _ = precision_recall_fscore_support(y_true, binary_opt, average="binary", zero_division=0)

    acc_05 = accuracy_score(y_true, binary_05)
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(y_true, binary_05, average="binary", zero_division=0)

    cm_opt = confusion_matrix(y_true, binary_opt).tolist()
    cm_05 = confusion_matrix(y_true, binary_05).tolist()

    rel_indices = torch.argmax(rel_tensors, dim=1).numpy()
    rel_names = {0: "Father-Daughter (FD)", 1: "Father-Son (FS)", 2: "Mother-Daughter (MD)", 3: "Mother-Son (MS)"}
    per_rel = {}
    for r_idx, r_name in rel_names.items():
        mask = (rel_indices == r_idx)
        if np.sum(mask) > 0:
            rel_acc = float(accuracy_score(y_true[mask], binary_opt[mask]) * 100)
            rel_count = int(np.sum(mask))
            per_rel[r_name] = {"accuracy": rel_acc, "sample_count": rel_count}

    return {
        "n_samples": len(y_true),
        "roc_auc": float(roc_auc),
        "optimal_threshold": opt_thresh,
        "optimal_accuracy": float(acc_opt * 100),
        "optimal_precision": float(prec_opt * 100),
        "optimal_recall": float(rec_opt * 100),
        "optimal_f1_score": float(f1_opt * 100),
        "default_05_accuracy": float(acc_05 * 100),
        "default_05_precision": float(prec_05 * 100),
        "default_05_recall": float(rec_05 * 100),
        "default_05_f1_score": float(f1_05 * 100),
        "confusion_matrix_optimal": cm_opt,
        "confusion_matrix_default": cm_05,
        "per_relation_breakdown": per_rel,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "preds": preds.tolist(),
        "y_true": y_true.tolist()
    }

def main():
    print("=================================================================")
    print("META-ENSEMBLE EVALUATION ACROSS ALL 4 DATASETS")
    print("=================================================================")

    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("\n[1/5] Loading Meta-Ensemble Classifier...")
    model = load_meta_ensemble(project_root)

    print("\n[2/5] Loading All 4 Datasets & Embedding Caches...")

    # Shared cache
    cache_path = os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    norm_cache = {os.path.normcase(os.path.abspath(k)): v for k, v in cache.items()}

    # 1. KinFaceW-I
    print("  Loading KinFaceW-I...")
    k1_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-I"))
    k1_t = prepare_pair_tensors(k1_pairs, norm_cache)

    # 2. KinFaceW-II
    print("  Loading KinFaceW-II...")
    k2_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-II"))
    k2_t = prepare_pair_tensors(k2_pairs, norm_cache)

    # 3. TSKinFace
    print("  Loading TSKinFace...")
    ts_path = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
    ts_pairs = load_tskinface_pairs(ts_path)
    ts_t = prepare_pair_tensors(ts_pairs, norm_cache)

    # 4. FIW Dataset
    print("  Loading Families In the Wild (FIW)...")
    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    with open(fiw_cache_path, "rb") as f:
        fiw_cache = pickle.load(f)
    norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}

    fiw_root = os.path.join(project_root, "public")
    fiw_pairs = load_fiw_pairs(fiw_root, max_pairs=500)
    fiw_t = prepare_pair_tensors(fiw_pairs, norm_fiw)

    dataset_tensors = {
        "KinFaceW-I Benchmark": k1_t,
        "KinFaceW-II Benchmark": k2_t,
        "TSKinFace Dataset": ts_t,
        "Families In the Wild (FIW)": fiw_t
    }

    print("\n[3/5] Computing Model Predictions on all 4 Datasets...")
    eval_results = {}
    for d_name, d_tensors in dataset_tensors.items():
        print(f"  Evaluating Meta-Ensemble on {d_name} ({len(d_tensors[2])} pairs)...")
        p = predict_batched(model, d_tensors[0], d_tensors[1], d_tensors[3])
        y = d_tensors[2].view(-1).numpy()
        eval_results[d_name] = evaluate_dataset(y, p, d_tensors[3])

    # Save JSON File
    json_path = os.path.join(results_dir, "meta_ensemble_4_datasets_comprehensive_metrics.json")
    json_clean = {}
    for d_name, res in eval_results.items():
        res_copy = {k: v for k, v in res.items() if k not in ["fpr", "tpr", "preds", "y_true"]}
        json_clean[d_name] = res_copy

    with open(json_path, "w") as f:
        json.dump(json_clean, f, indent=2)
    print(f"\n[SAVED JSON] {json_path}")

    print("\n[4/5] Rendering Publication Quality Plots...")

    # -------------------------------------------------------------------------
    # PLOT 1: ROC Curves Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for d_name, res in eval_results.items():
        ax.plot(res["fpr"], res["tpr"], label=f"{d_name} (AUC = {res['roc_auc']:.4f})", color=DATASET_COLORS[d_name], lw=2.2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Random Chance (AUC = 0.50)")
    ax.set_title("Meta-Ensemble ROC Curves Across 4 Major Datasets", fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.4)

    p1_path1 = os.path.join(plots_dir, "meta_ensemble_4_datasets_roc_curves.png")
    p1_path2 = os.path.join(brain_dir, "meta_ensemble_4_datasets_roc_curves.png")
    plt.savefig(p1_path1)
    plt.savefig(p1_path2)
    plt.close()
    print(f"  [SAVED PLOT] {p1_path1}")

    # -------------------------------------------------------------------------
    # PLOT 2: Multi-Metric Comparison Bar Chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    d_names = list(eval_results.keys())
    x = np.arange(len(d_names))
    width = 0.18

    auc_vals = [eval_results[d]["roc_auc"] * 100 for d in d_names]
    acc_vals = [eval_results[d]["optimal_accuracy"] for d in d_names]
    prec_vals = [eval_results[d]["optimal_precision"] for d in d_names]
    rec_vals = [eval_results[d]["optimal_recall"] for d in d_names]

    rects1 = ax.bar(x - 1.5*width, auc_vals, width, label="ROC-AUC (%)", color="#3F51B5")
    rects2 = ax.bar(x - 0.5*width, acc_vals, width, label="Optimal Accuracy (%)", color="#4CAF50")
    rects3 = ax.bar(x + 0.5*width, prec_vals, width, label="Precision (%)", color="#FF9800")
    rects4 = ax.bar(x + 1.5*width, rec_vals, width, label="Recall (%)", color="#E91E63")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Meta-Ensemble Metric Overview Across 4 Kinship Datasets", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" Dataset", "").replace(" Benchmark", "") for d in d_names], fontweight="bold")
    ax.set_ylim(40, 100)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar heights
    for rects in [rects1, rects2, rects3, rects4]:
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    p2_path1 = os.path.join(plots_dir, "meta_ensemble_4_datasets_metrics_bar_chart.png")
    p2_path2 = os.path.join(brain_dir, "meta_ensemble_4_datasets_metrics_bar_chart.png")
    plt.savefig(p2_path1)
    plt.savefig(p2_path2)
    plt.close()
    print(f"  [SAVED PLOT] {p2_path1}")

    # -------------------------------------------------------------------------
    # PLOT 3: Per-Relation Breakdown Across Datasets
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rel_keys = ["Father-Daughter (FD)", "Father-Son (FS)", "Mother-Daughter (MD)", "Mother-Son (MS)"]
    rel_labels = ["Father-Daughter\n(FD)", "Father-Son\n(FS)", "Mother-Daughter\n(MD)", "Mother-Son\n(MS)"]

    x_rel = np.arange(len(rel_keys))
    w_bar = 0.18

    for idx, d in enumerate(d_names):
        rel_dict = eval_results[d]["per_relation_breakdown"]
        accs = [rel_dict.get(r, {}).get("accuracy", 0.0) for r in rel_keys]
        bars = ax.bar(x_rel + (idx - 1.5)*w_bar, accs, w_bar, label=d.replace(" Dataset", "").replace(" Benchmark", ""), color=DATASET_COLORS[d])
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Per-Relation Kinship Accuracy Breakdown Across 4 Datasets", fontweight="bold", pad=12)
    ax.set_xticks(x_rel)
    ax.set_xticklabels(rel_labels, fontweight="bold")
    ax.set_ylim(45, 100)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    p3_path1 = os.path.join(plots_dir, "meta_ensemble_4_datasets_relation_breakdown.png")
    p3_path2 = os.path.join(brain_dir, "meta_ensemble_4_datasets_relation_breakdown.png")
    plt.savefig(p3_path1)
    plt.savefig(p3_path2)
    plt.close()
    print(f"  [SAVED PLOT] {p3_path1}")

    # -------------------------------------------------------------------------
    # PLOT 4: 2x2 Grid of Confusion Matrices
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    for idx, d in enumerate(d_names):
        ax = axes[idx]
        cm = np.array(eval_results[d]["confusion_matrix_optimal"])
        
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{d.replace(' Dataset', '').replace(' Benchmark', '')}\n(Acc: {eval_results[d]['optimal_accuracy']:.1f}%, AUC: {eval_results[d]['roc_auc']:.3f})", fontweight="bold", fontsize=10)
        
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Non-Kin", "Kin"], fontweight="bold")
        ax.set_yticklabels(["Non-Kin", "Kin"], fontweight="bold")

        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, f"{val}\n({val/cm.sum()*100:.1f}%)", ha="center", va="center", color="white" if val > thresh else "black", fontweight="bold")

        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_xlabel("Predicted Label", fontweight="bold")

    plt.tight_layout()
    p4_path1 = os.path.join(plots_dir, "meta_ensemble_4_datasets_confusion_matrices.png")
    p4_path2 = os.path.join(brain_dir, "meta_ensemble_4_datasets_confusion_matrices.png")
    plt.savefig(p4_path1)
    plt.savefig(p4_path2)
    plt.close()
    print(f"  [SAVED PLOT] {p4_path1}")

    print("\n=================================================================")
    print("[SUCCESS] EVALUATION COMPLETED ACROSS ALL 4 DATASETS!")
    print("=================================================================")

if __name__ == "__main__":
    main()
