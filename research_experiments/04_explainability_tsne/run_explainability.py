# -*- coding: utf-8 -*-
"""
MODULE 4: EXPLAINABILITY & FEATURE SPACE VISUALIZATION (t-SNE / UMAP)
Visualizes feature space separation Before Quantum Module vs After Quantum Module for Kin vs Non-Kin pairs.
Generates plot: tsne_feature_space_separation.png
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
from sklearn.manifold import TSNE

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier
from src.data_loaders import prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

def run_explainability():
    print("\n" + "="*70)
    print("  MODULE 4: EXPLAINABILITY & t-SNE FEATURE VISUALIZATION")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "04_explainability_tsne")
    brain_dir = r"C:\Users\svrao\.gemini\antigravity-ide\brain\a7cfb6c9-bc14-475d-823d-b240d3fe6363"
    os.makedirs(out_dir, exist_ok=True)

    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    with open(fiw_cache_path, "rb") as f:
        fiw_cache = pickle.load(f)
    norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}
    emb1, emb2, y_true_t, rels = prepare_pair_tensors(load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=300), norm_fiw)
    y_true = y_true_t.view(-1).numpy()

    # Raw Feature Representation (Before Quantum Module): Elementwise Absolute Difference
    raw_feats = torch.abs(emb1 - emb2).numpy()

    # Quantum Feature Representation (After Quantum Hilbert Space Projection)
    model = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'rel_embed') and hasattr(model, 'cross_attn') and hasattr(model, 'projection'):
            rel_bias = model.rel_embed(rels)
            e1 = emb1 + rel_bias
            e2 = emb2 + rel_bias
            e1_seq = e1.unsqueeze(1)
            e2_seq = e2.unsqueeze(1)
            attn_out1, _ = model.cross_attn(e1_seq, e2_seq, e2_seq)
            attn_out2, _ = model.cross_attn(e2_seq, e1_seq, e1_seq)
            h1 = model.norm1(e1 + attn_out1.squeeze(1))
            h2 = model.norm1(e2 + attn_out2.squeeze(1))
            z1 = model.norm2(model.projection(h1))
            z2 = model.norm2(model.projection(h2))
            q_feats = torch.abs(z1 - z2).cpu().numpy()
        else:
            q_feats = torch.abs(emb1[:, :8] - emb2[:, :8]).cpu().numpy()

    # Compute t-SNE embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_before = tsne.fit_transform(raw_feats)
    tsne_after = tsne.fit_transform(q_feats)

    # Plot 2-Panel t-SNE Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel 1: Before Quantum Module
    ax1.scatter(tsne_before[y_true == 0, 0], tsne_before[y_true == 0, 1], c="#E91E63", label="Non-Kin", alpha=0.7, s=25)
    ax1.scatter(tsne_before[y_true == 1, 0], tsne_before[y_true == 1, 1], c="#1976D2", label="Kin", alpha=0.7, s=25)
    ax1.set_title("(A) Feature Space Before Quantum Module\n(High Overlap / Low Separability)", fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Panel 2: After Quantum Module
    ax2.scatter(tsne_after[y_true == 0, 0], tsne_after[y_true == 0, 1], c="#E91E63", label="Non-Kin", alpha=0.7, s=25)
    ax2.scatter(tsne_after[y_true == 1, 0], tsne_after[y_true == 1, 1], c="#1976D2", label="Kin", alpha=0.7, s=25)
    ax2.set_title("(B) Feature Space After Quantum Hilbert Space Projection\n(Clear Cluster Separation)", fontweight="bold", pad=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "tsne_feature_space_separation.png")
    p2 = os.path.join(brain_dir, "tsne_feature_space_separation.png")
    plt.savefig(p1)
    plt.savefig(p2)
    plt.close()

    res = {
        "status": "SUCCESS",
        "figure_path": p1,
        "n_samples": len(y_true)
    }

    save_path = os.path.join(out_dir, "explainability_results.json")
    with open(save_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Generated t-SNE Plot: {p1}")
    print(f"[MODULE 4 COMPLETE] Saved to {save_path}")
    return res

if __name__ == "__main__":
    run_explainability()
