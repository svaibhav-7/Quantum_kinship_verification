# -*- coding: utf-8 -*-
"""
MODULE 05: ROC & PR CURVES (per model)

For each of the 4 models, computes ROC and PR curves across the 4 datasets.
Produces per-model JSON + per-model ROC/PR png + combined plots.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets, get_predictions
from common_utils import DATASET_COLORS, save_json

OUT_DIR = os.path.join(_module_root, "05_roc_pr_curves")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_NAMES = ["KinFaceW-I", "KinFaceW-II", "TSKinFace", "FIW"]


def run_roc_pr():
    print("\n" + "=" * 70)
    print("  MODULE 05: ROC & PR CURVES (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()

    all_results = {}
    for mkey in MODEL_KEYS:
        model = models[mkey]
        metrics = {}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        for dname in DATASET_NAMES:
            emb1, emb2, y_true_t, rels, _ = datasets[dname]
            y_true = y_true_t.view(-1).numpy()
            preds = get_predictions(model, emb1, emb2, rels)
            if len(y_true) == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true, preds)
            roc_auc = float(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_true, preds)
            pr_auc = float(average_precision_score(y_true, preds))

            metrics[dname] = {"roc_auc": roc_auc, "pr_auc": pr_auc}
            ax1.plot(fpr, tpr, label=f"{dname} (AUC={roc_auc:.4f})", color=DATASET_COLORS[dname], lw=2)
            ax2.plot(rec, prec, label=f"{dname} (PR-AUC={pr_auc:.4f})", color=DATASET_COLORS[dname], lw=2)

        ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
        ax1.set_title(f"ROC Curves - {MODEL_LABELS[mkey]}", fontweight="bold")
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.legend(loc="lower right", fontsize=7); ax1.grid(True, linestyle="--", alpha=0.3)
        ax2.set_title(f"PR Curves - {MODEL_LABELS[mkey]}", fontweight="bold")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.legend(loc="lower left", fontsize=7); ax2.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"roc_pr_{mkey}.png"))
        plt.close()

        save_json(metrics, os.path.join(OUT_DIR, f"{mkey}.json"))
        all_results[mkey] = metrics
        print(f"  [MODEL {mkey}] {metrics}")

    _plot_combined(all_results, datasets, models)
    save_json(all_results, os.path.join(OUT_DIR, "roc_pr_summary.json"))
    print("  [MODULE 05 COMPLETE]")
    return all_results


def _plot_combined(all_results, datasets, models):
    # Combined ROC: plot each model's ROC for FIW
    emb1, emb2, y_true_t, rels, _ = datasets["FIW"]
    y_true = y_true_t.view(-1).numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    mcolors = {"ensemble_kinship_full": "#E91E63", "meta_ensemble_kinship": "#3F51B5",
               "ensemble_kinship_fiw": "#009688", "best_checkpoint": "#FF9800"}
    for mkey in MODEL_KEYS:
        preds = get_predictions(models[mkey], emb1, emb2, rels)
        fpr, tpr, _ = roc_curve(y_true, preds)
        ax.plot(fpr, tpr, label=f"{MODEL_LABELS[mkey]} (AUC={auc(fpr, tpr):.4f})", color=mcolors[mkey], lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    ax.set_title("Combined ROC Curves on FIW - All 4 Models", fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_combined_fiw_4models.png"))
    plt.close()


if __name__ == "__main__":
    run_roc_pr()
