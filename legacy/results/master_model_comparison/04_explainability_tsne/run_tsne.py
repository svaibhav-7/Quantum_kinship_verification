# -*- coding: utf-8 -*-
"""
MODULE 04: EXPLAINABILITY & t-SNE FEATURE VISUALIZATION (per model)

For each of the 4 models, computes t-SNE of the raw feature representation
(absolute difference of embeddings) vs. the post-quantum feature representation
(absolute difference of projected angles) for Kin vs Non-Kin pairs.
Produces per-model PNG + a combined plot.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets

OUT_DIR = os.path.join(_module_root, "04_explainability_tsne")
os.makedirs(OUT_DIR, exist_ok=True)


def run_tsne():
    print("\n" + "=" * 70)
    print("  MODULE 04: EXPLAINABILITY & t-SNE (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use FIW for visualization
    emb1, emb2, y_true_t, rels, _ = datasets["FIW"]
    y_true = y_true_t.view(-1).numpy()
    n = min(len(y_true), 300)
    emb1, emb2, rels, y_true = emb1[:n], emb2[:n], rels[:n], y_true[:n]

    raw_feats = torch.abs(emb1 - emb2).numpy()

    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(raw_feats)

    results = {}
    for mkey in MODEL_KEYS:
        model = models[mkey].to(device)
        # Get projected angles via the projection_net of the first sub-model for ensembles
        q_feats = _get_quantum_features(model, emb1, emb2, rels, device)
        tsne_q = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(q_feats)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
        _plot_panel(ax1, tsne_raw, y_true, "(A) Before Quantum Module")
        _plot_panel(ax2, tsne_q, y_true, "(B) After Quantum Hilbert Projection")
        fig.suptitle(f"t-SNE - {MODEL_LABELS[mkey]}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"tsne_{mkey}.png"))
        plt.close()

        gap = _compute_gap(q_feats, y_true)
        results[mkey] = {"model": MODEL_LABELS[mkey], "n": int(n),
                         "quantum_separation_gap": float(gap), "figure": f"tsne_{mkey}.png"}
        print(f"  [MODEL {mkey}] gap={gap:.4f}")

    _plot_combined(tsne_raw, models, emb1, emb2, rels, y_true, device)

    import json
    with open(os.path.join(OUT_DIR, "explainability_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  [MODULE 04 COMPLETE]")
    return results


def _get_quantum_features(model, emb1, emb2, rels, device):
    # Access the projection_net; for ensembles, use first sub-model's projection_net
    proj = None
    candidates = []
    candidates.append(model)
    md_list = getattr(model, "models", None)
    if md_list is not None:
        candidates.extend(md_list)
    enum_full = getattr(getattr(model, "ensemble_full", None), "models", None)
    if enum_full is not None:
        candidates.extend(enum_full)
    candidates.append(getattr(model, "single_fiw", None))

    for m in candidates:
        if m is not None and hasattr(getattr(m, "projection_net", None), "projection"):
            proj = m.projection_net
            break
    if proj is None:
        raise RuntimeError("Could not find projection_net")
    proj = proj.to(device)
    with torch.no_grad():
        z1, z2 = proj(emb1.to(device), emb2.to(device), rels.to(device))
    return torch.abs(z1 - z2).cpu().numpy()


def _plot_panel(ax, tsne_xy, y_true, title):
    ax.scatter(tsne_xy[y_true == 0, 0], tsne_xy[y_true == 0, 1], c="#E91E63", label="Non-Kin", alpha=0.7, s=25)
    ax.scatter(tsne_xy[y_true == 1, 0], tsne_xy[y_true == 1, 1], c="#1976D2", label="Kin", alpha=0.7, s=25)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)


def _compute_gap(q_feats, y_true):
    if np.any(y_true == 1) and np.any(y_true == 0):
        return float(np.mean(q_feats[y_true == 1].sum(1)) - np.mean(q_feats[y_true == 0].sum(1)))
    return 0.0


def _plot_combined(tsne_raw, models, emb1, emb2, rels, y_true, device):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, mkey in enumerate(MODEL_KEYS):
        model = models[mkey].to(device)
        q_feats = _get_quantum_features(model, emb1, emb2, rels, device)
        tsne_q = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(q_feats)
        ax = axes[i]
        ax.scatter(tsne_q[y_true == 0, 0], tsne_q[y_true == 0, 1], c="#E91E63", label="Non-Kin", alpha=0.7, s=20)
        ax.scatter(tsne_q[y_true == 1, 0], tsne_q[y_true == 1, 1], c="#1976D2", label="Kin", alpha=0.7, s=20)
        ax.set_title(MODEL_LABELS[mkey], fontweight="bold")
        ax.legend(loc="upper right", frameon=True, fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("t-SNE After Quantum Projection - All 4 Models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_combined_4models.png"))
    plt.close()


if __name__ == "__main__":
    run_tsne()
