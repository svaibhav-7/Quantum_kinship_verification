# -*- coding: utf-8 -*-
"""
MODULE 5: COMPREHENSIVE ROC AND PRECISION-RECALL (PR) CURVES
Generates publication-quality ROC and PR Curves for all 4 datasets.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import load_kinfacew_pairs, load_tskinface_pairs, prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

DATASET_COLORS = {
    "KinFaceW-I": "#1976D2",
    "KinFaceW-II": "#009688",
    "TSKinFace": "#FF9800",
    "FIW": "#9C27B0"
}

def run_roc_pr_curves():
    print("\n" + "="*70)
    print("  MODULE 5: COMPREHENSIVE ROC AND PRECISION-RECALL (PR) CURVES")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "05_roc_pr_curves")
    brain_dir = r"C:\Users\svrao\.gemini\antigravity-ide\brain\a7cfb6c9-bc14-475d-823d-b240d3fe6363"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Datasets
    cache_path = os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    norm_cache = {os.path.normcase(os.path.abspath(k)): v for k, v in cache.items()}

    k1_t = prepare_pair_tensors(load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-I")), norm_cache)
    k2_t = prepare_pair_tensors(load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-II")), norm_cache)
    ts_t = prepare_pair_tensors(load_tskinface_pairs(os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")), norm_cache)

    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    with open(fiw_cache_path, "rb") as f:
        fiw_cache = pickle.load(f)
    norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}
    fiw_t = prepare_pair_tensors(load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500), norm_fiw)

    datasets = {
        "KinFaceW-I": k1_t,
        "KinFaceW-II": k2_t,
        "TSKinFace": ts_t,
        "FIW": fiw_t
    }

    # Load Model
    meta_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    
    full_meta = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
    full_meta.load_state_dict(torch.load(meta_path, map_location="cpu"))
    full_meta.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_meta.to(device)

    # 2. Render 2-Panel Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    metrics_out = {}
    for d_name, d_tensors in datasets.items():
        emb1, emb2, y_true_t, rels = d_tensors
        y_true = y_true_t.view(-1).numpy()
        with torch.no_grad():
            preds = full_meta(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()

        fpr, tpr, _ = roc_curve(y_true, preds)
        roc_auc = float(auc(fpr, tpr))

        precision, recall, _ = precision_recall_curve(y_true, preds)
        pr_auc = float(average_precision_score(y_true, preds))

        ax1.plot(fpr, tpr, label=f"{d_name} (ROC-AUC = {roc_auc:.4f})", color=DATASET_COLORS[d_name], lw=2)
        ax2.plot(recall, precision, label=f"{d_name} (PR-AUC = {pr_auc:.4f})", color=DATASET_COLORS[d_name], lw=2)

        metrics_out[d_name] = {"roc_auc": roc_auc, "pr_auc": pr_auc}

    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Chance")
    ax1.set_title("ROC Curves Across 4 Datasets", fontweight="bold", pad=10)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_title("Precision-Recall (PR) Curves Across 4 Datasets", fontweight="bold", pad=10)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "roc_pr_curves_combined.png")
    p2 = os.path.join(brain_dir, "roc_pr_curves_combined.png")
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()

    save_path = os.path.join(out_dir, "roc_pr_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Generated ROC & PR Curves Plot: {p1}")
    print(f"[MODULE 5 COMPLETE] Saved to {save_path}")
    return metrics_out

if __name__ == "__main__":
    run_roc_pr_curves()
