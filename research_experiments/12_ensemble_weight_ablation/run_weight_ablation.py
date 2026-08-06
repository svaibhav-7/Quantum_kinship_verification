# -*- coding: utf-8 -*-
"""
MODULE 12: ENSEMBLE FUSION WEIGHT ABLATION
Evaluates Equal Weighting (0.33, 0.33, 0.33) vs Single Domain Weighting vs Learned Optimal Weighting (0.45, 0.35, 0.20).
Generates plot: fusion_weights_ablation_chart.png
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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

def run_weight_ablation():
    print("\n" + "="*70)
    print("  MODULE 12: ENSEMBLE FUSION WEIGHT ABLATION STUDY")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "12_ensemble_weight_ablation")
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
    
    meta_state = torch.load(meta_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusion_configs = {
        "Equal Weighting\n(0.33, 0.33, 0.33)": (0.333, 0.333, 0.334),
        "Base-Dominant\n(0.70, 0.20, 0.10)": (0.70, 0.20, 0.10),
        "FIW-Dominant\n(0.20, 0.70, 0.10)": (0.20, 0.70, 0.10),
        "Learned Optimal\n(0.45, 0.35, 0.20)": (0.45, 0.35, 0.20)
    }

    weight_results = {}

    for name, weights in fusion_configs.items():
        model = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=weights)
        model.load_state_dict(meta_state)
        model.eval()
        model.to(device)

        with torch.no_grad():
            preds = model(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()

        acc = float(accuracy_score(y_true, preds >= 0.5216) * 100)
        auc_score = float(roc_auc_score(y_true, preds) * 100)
        f1 = float(f1_score(y_true, preds >= 0.5216) * 100)

        clean_name = name.replace("\n", " ")
        weight_results[clean_name] = {
            "weights": list(weights),
            "accuracy": acc,
            "roc_auc": auc_score,
            "f1_score": f1
        }

    # Plot Bar Chart
    names = list(fusion_configs.keys())
    accs = [weight_results[n.replace("\n", " ")]["accuracy"] for n in names]
    aucs = [weight_results[n.replace("\n", " ")]["roc_auc"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    w = 0.35

    r1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#009688")
    r2 = ax.bar(x + w/2, aucs, w, label="ROC-AUC (%)", color="#9C27B0")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Meta-Ensemble Domain Fusion Weighting Strategy Ablation", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold", fontsize=9)
    ax.set_ylim(65, 92)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for r in r1 + r2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "fusion_weights_ablation_chart.png")
    p2 = os.path.join(brain_dir, "fusion_weights_ablation_chart.png")
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()

    save_path = os.path.join(out_dir, "fusion_weight_ablation.json")
    with open(save_path, "w") as f:
        json.dump(weight_results, f, indent=2)

    print(f"Generated Plot: {p1}")
    print(f"[MODULE 12 COMPLETE] Saved to {save_path}")
    return weight_results

if __name__ == "__main__":
    run_weight_ablation()
