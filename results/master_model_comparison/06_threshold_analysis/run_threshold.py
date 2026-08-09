# -*- coding: utf-8 -*-
"""
MODULE 06: THRESHOLD SENSITIVITY ANALYSIS (per model)

For each of the 4 models, sweeps decision thresholds on the FIW dataset and
records Accuracy/Precision/Recall/F1 curves. Produces per-model JSON + png
and a combined plot.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets, get_predictions
from common_utils import save_json

OUT_DIR = os.path.join(_module_root, "06_threshold_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLDS = np.linspace(0.30, 0.70, 50)


def run_threshold():
    print("\n" + "=" * 70)
    print("  MODULE 06: THRESHOLD SENSITIVITY ANALYSIS (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()
    emb1, emb2, y_true_t, rels, _ = datasets["FIW"]
    y_true = y_true_t.view(-1).numpy()

    all_results = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    mcolors = {"ensemble_kinship_full": "#E91E63", "meta_ensemble_kinship": "#3F51B5",
               "ensemble_kinship_fiw": "#009688", "best_checkpoint": "#FF9800"}

    for mkey in MODEL_KEYS:
        preds = get_predictions(models[mkey], emb1, emb2, rels)
        accs, precs, recs, f1s = [], [], [], []
        for t in THRESHOLDS:
            b = (preds >= t).astype(int)
            accs.append(accuracy_score(y_true, b) * 100)
            precs.append(precision_score(y_true, b, zero_division=0) * 100)
            recs.append(recall_score(y_true, b, zero_division=0) * 100)
            f1s.append(f1_score(y_true, b, zero_division=0) * 100)

        opt_idx = int(np.argmax(accs))
        res = {
            "optimal_threshold": float(THRESHOLDS[opt_idx]),
            "max_accuracy": float(accs[opt_idx]),
            "precision_at_opt": float(precs[opt_idx]),
            "recall_at_opt": float(recs[opt_idx]),
            "f1_at_opt": float(f1s[opt_idx]),
            "threshold_curve": {
                "thresholds": THRESHOLDS.tolist(),
                "accuracy": accs, "precision": precs, "recall": recs, "f1": f1s,
            },
        }
        save_json(res, os.path.join(OUT_DIR, f"{mkey}.json"))
        all_results[mkey] = res
        ax.plot(THRESHOLDS, accs, label=f"{MODEL_LABELS[mkey]} (Acc)", color=mcolors[mkey], lw=2)
        print(f"  [MODEL {mkey}] opt_t={res['optimal_threshold']:.3f}, acc={res['max_accuracy']:.2f}%")

    ax.set_xlabel("Decision Threshold (tau)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Threshold Sensitivity - Accuracy vs Threshold (FIW)", fontweight="bold")
    ax.legend(loc="lower center", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "threshold_accuracy_4models.png"))
    plt.close()

    save_json(all_results, os.path.join(OUT_DIR, "threshold_summary.json"))
    print("  [MODULE 06 COMPLETE]")
    return all_results


if __name__ == "__main__":
    run_threshold()
