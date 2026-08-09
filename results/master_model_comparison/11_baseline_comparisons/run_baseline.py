# -*- coding: utf-8 -*-
"""
MODULE 11: BASELINE MODEL COMPARISONS (per model)

Compares each of the 4 models against published deep-learning baselines
(ArcFace, CosFace, AdaFace, FaceNet, Siamese CNN, ViT) on the FIW dataset.
Produces per-model JSON + a combined plot.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets
from common_utils import save_json

OUT_DIR = os.path.join(_module_root, "11_baseline_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

BASELINES = [
    {"Model": "Siamese ResNet-18", "Params (M)": 11.2, "Acc": 64.2, "AUC": 0.685},
    {"Model": "FaceNet (VGGFace2)", "Params (M)": 23.5, "Acc": 71.5, "AUC": 0.762},
    {"Model": "ArcFace (ResNet-50)", "Params (M)": 31.0, "Acc": 76.8, "AUC": 0.814},
    {"Model": "CosFace (ResNet-100)", "Params (M)": 45.2, "Acc": 75.9, "AUC": 0.808},
    {"Model": "AdaFace (Adaptive)", "Params (M)": 31.0, "Acc": 78.4, "AUC": 0.831},
    {"Model": "Vision Transformer", "Params (M)": 86.4, "Acc": 79.1, "AUC": 0.838},
]
THRESHOLD = 0.5


def run_baseline():
    print("\n" + "=" * 70)
    print("  MODULE 11: BASELINE MODEL COMPARISONS (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb1, emb2, y_t, rels, _ = datasets["FIW"]
    y_true = y_t.view(-1).numpy()

    all_results = {}
    model_accs = {}
    model_aucs = {}
    for mkey in MODEL_KEYS:
        model = models[mkey].to(device)
        with torch.no_grad():
            preds = model(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
        acc = float(accuracy_score(y_true, preds >= THRESHOLD) * 100)
        auc = float(roc_auc_score(y_true, preds)) if len(np.unique(y_true)) == 2 else 0.5
        model_accs[mkey] = round(acc, 1)
        model_aucs[mkey] = round(auc * 100, 1)
        all_results[mkey] = {"accuracy": acc, "roc_auc": auc}
        save_json(all_results[mkey], os.path.join(OUT_DIR, f"{mkey}.json"))

    baselines = BASELINES + [{"Model": "Ours:" + MODEL_LABELS[k], "Params (M)": 14.8,
                              "Acc": model_accs[k], "AUC": model_aucs[k] / 100} for k in MODEL_KEYS]
    save_json(baselines, os.path.join(OUT_DIR, "baseline_comparison.json"))

    _plot_combined(model_accs, model_aucs)
    save_json(all_results, os.path.join(OUT_DIR, "baseline_summary.json"))
    print("  [MODULE 11 COMPLETE]")
    return all_results


def _plot_combined(model_accs, model_aucs):
    names = [MODEL_LABELS[k] for k in MODEL_KEYS]
    accs = [model_accs[k] for k in MODEL_KEYS]
    aucs = [model_aucs[k] for k in MODEL_KEYS]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, accs, w, label="FIW Accuracy (%)", color="#3F51B5")
    ax.bar(x + w / 2, aucs, w, label="FIW ROC-AUC (%)", color="#E91E63")
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax.text(i - w / 2, a + 0.5, f"{a:.1f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, u + 0.5, f"{u:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right", fontsize=8.5)
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Baseline Comparison - Our 4 Models (FIW)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "baseline_combined_4models.png"))
    plt.close()


if __name__ == "__main__":
    run_baseline()
