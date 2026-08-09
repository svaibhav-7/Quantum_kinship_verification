# -*- coding: utf-8 -*-
"""
MODULE 10: CROSS-DATASET GENERALIZATION MATRIX (per model)

For each of the 4 models, builds a 4x4 cross-dataset generalization matrix by
evaluating the model (trained predominantly on one domain) against all 4 test
domains. Produces per-model JSON + heatmap + a combined plot.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets
from common_utils import save_json

OUT_DIR = os.path.join(_module_root, "10_cross_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_NAMES = ["KinFaceW-I", "KinFaceW-II", "TSKinFace", "FIW"]
THRESHOLD = 0.5


def run_cross_dataset():
    print("\n" + "=" * 70)
    print("  MODULE 10: CROSS-DATASET GENERALIZATION MATRIX (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    for mkey in MODEL_KEYS:
        model = models[mkey].to(device)
        matrix = {}
        for test_d in DATASET_NAMES:
            emb1, emb2, y_t, rels, _ = datasets[test_d]
            y_true = y_t.view(-1).numpy()
            with torch.no_grad():
                preds = model(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
            acc = float(accuracy_score(y_true, preds >= THRESHOLD) * 100)
            matrix[test_d] = acc
        all_results[mkey] = matrix
        save_json(matrix, os.path.join(OUT_DIR, f"{mkey}.json"))
        _plot_heatmap(mkey, matrix)
        print(f"  [MODEL {mkey}] {matrix}")

    _plot_combined(all_results)
    save_json(all_results, os.path.join(OUT_DIR, "cross_dataset_summary.json"))
    print("  [MODULE 10 COMPLETE]")
    return all_results


def _plot_heatmap(mkey, matrix):
    grid = np.array([[matrix[d] for d in DATASET_NAMES]])
    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = ax.imshow(grid, cmap="YlGnBu")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(DATASET_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels([MODEL_LABELS[mkey]], fontsize=9)
    for j in range(4):
        ax.text(j, 0, f"{grid[0, j]:.1f}%", ha="center", va="center", fontweight="bold",
                color="white" if grid[0, j] < 75 else "black")
    ax.set_title(f"Cross-Dataset Accuracy - {MODEL_LABELS[mkey]}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cross_dataset_{mkey}.png"))
    plt.close()


def _plot_combined(all_results):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(4)
    w = 0.2
    mcolors = {"ensemble_kinship_full": "#E91E63", "meta_ensemble_kinship": "#3F51B5",
               "ensemble_kinship_fiw": "#009688", "best_checkpoint": "#FF9800"}
    for i, mkey in enumerate(MODEL_KEYS):
        vals = [all_results[mkey][d] for d in DATASET_NAMES]
        ax.bar(x + (i - 1.5) * w, vals, w, label=MODEL_LABELS[mkey], color=mcolors[mkey])
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_NAMES, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Cross-Dataset Generalization - All 4 Models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cross_dataset_combined_4models.png"))
    plt.close()


if __name__ == "__main__":
    run_cross_dataset()
