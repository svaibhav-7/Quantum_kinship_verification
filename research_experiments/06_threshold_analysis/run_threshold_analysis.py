# -*- coding: utf-8 -*-
"""
MODULE 6: THRESHOLD SENSITIVITY AND ROBUSTNESS ANALYSIS
Evaluates Accuracy, Precision, Recall, and F1 score continuous curves across decision thresholds (tau in [0.30, 0.70]).
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

def run_threshold_analysis():
    print("\n" + "="*70)
    print("  MODULE 6: THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "06_threshold_analysis")
    brain_dir = r"C:\Users\svrao\.gemini\antigravity-ide\brain\a7cfb6c9-bc14-475d-823d-b240d3fe6363"
    os.makedirs(out_dir, exist_ok=True)

    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    with open(fiw_cache_path, "rb") as f:
        fiw_cache = pickle.load(f)
    norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}
    emb1, emb2, y_true_t, rels = prepare_pair_tensors(load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500), norm_fiw)
    y_true = y_true_t.view(-1).numpy()

    meta_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    
    full_meta = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
    full_meta.load_state_dict(torch.load(meta_path, map_location="cpu"))
    full_meta.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_meta.to(device)

    with torch.no_grad():
        preds = full_meta(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()

    thresholds = np.linspace(0.30, 0.70, 50)
    accs, precs, recs, f1s = [], [], [], []

    for t in thresholds:
        b = (preds >= t).astype(int)
        accs.append(accuracy_score(y_true, b) * 100)
        precs.append(precision_score(y_true, b, zero_division=0) * 100)
        recs.append(recall_score(y_true, b, zero_division=0) * 100)
        f1s.append(f1_score(y_true, b, zero_division=0) * 100)

    opt_idx = np.argmax(accs)
    opt_t = float(thresholds[opt_idx])

    # Plot Threshold Curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, accs, label="Accuracy (%)", color="#4CAF50", lw=2.2)
    ax.plot(thresholds, precs, label="Precision (%)", color="#FF9800", lw=2.2)
    ax.plot(thresholds, recs, label="Recall (%)", color="#E91E63", lw=2.2)
    ax.plot(thresholds, f1s, label="F1 Score (%)", color="#3F51B5", lw=2.2)

    ax.axvline(opt_t, color="black", linestyle="--", alpha=0.7, label=f"Optimal Threshold (tau = {opt_t:.4f})")
    ax.set_xlabel("Decision Threshold (tau)", fontweight="bold")
    ax.set_ylabel("Metric Score (%)", fontweight="bold")
    ax.set_title("Threshold Sensitivity & Performance Trade-Off Curves (FIW Dataset)", fontweight="bold", pad=12)
    ax.legend(loc="lower center", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "threshold_sensitivity_curves.png")
    p2 = os.path.join(brain_dir, "threshold_sensitivity_curves.png")
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()

    res = {
        "optimal_threshold": opt_t,
        "max_accuracy": float(accs[opt_idx]),
        "precision_at_opt": float(precs[opt_idx]),
        "recall_at_opt": float(recs[opt_idx]),
        "f1_at_opt": float(f1s[opt_idx])
    }

    save_path = os.path.join(out_dir, "threshold_metrics.json")
    with open(save_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Optimal Threshold: {opt_t:.4f} (Max Acc: {accs[opt_idx]:.2f}%)")
    print(f"[MODULE 6 COMPLETE] Saved to {save_path}")
    return res

if __name__ == "__main__":
    run_threshold_analysis()
