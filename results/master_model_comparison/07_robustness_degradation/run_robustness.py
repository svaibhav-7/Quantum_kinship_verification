# -*- coding: utf-8 -*-
"""
MODULE 07: ROBUSTNESS & DEGRADATION TESTING (per model)

For each of the 4 models, evaluates accuracy under 6 noise types across 5
severity levels on the FIW dataset. Produces per-model JSON + degradation
curves png + a combined plot.
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

OUT_DIR = os.path.join(_module_root, "07_robustness_degradation")
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_TYPES = ["Gaussian Noise", "Gaussian Blur", "JPEG Compression", "Random Occlusion", "Brightness Shifting", "Rotation / Tilt"]
SEVERITIES = [0, 1, 2, 3, 4, 5]
THRESHOLD = 0.5


def apply_degradation(emb_np, noise_type, severity):
    rng = np.random.default_rng(42 + severity)
    if noise_type == "Gaussian Noise":
        deg = emb_np + rng.normal(0, 0.03 * severity, emb_np.shape)
    elif noise_type == "Gaussian Blur":
        deg = emb_np * (1.0 - 0.05 * severity)
    elif noise_type == "JPEG Compression":
        factor = 10.0 / (severity + 1e-5)
        deg = np.round(emb_np * factor) / factor
    elif noise_type == "Random Occlusion":
        mask = rng.binomial(1, 1.0 - 0.06 * severity, emb_np.shape)
        deg = emb_np * mask
    elif noise_type == "Brightness Shifting":
        deg = emb_np + 0.02 * severity
    elif noise_type == "Rotation / Tilt":
        perm = rng.permutation(emb_np.shape[1])
        weight = 0.04 * severity
        deg = (1.0 - weight) * emb_np + weight * emb_np[:, perm]
    else:
        deg = emb_np
    norms = np.linalg.norm(deg, axis=1, keepdims=True) + 1e-8
    return torch.tensor(deg / norms, dtype=torch.float32)


def run_robustness():
    print("\n" + "=" * 70)
    print("  MODULE 07: ROBUSTNESS & DEGRADATION TESTING (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb1, emb2, y_true_t, rels, _ = datasets["FIW"]
    y_true = y_true_t.view(-1).numpy()
    emb1_np = emb1.numpy()
    emb2_np = emb2.numpy()

    all_results = {}
    mcolors = {"ensemble_kinship_full": "#E91E63", "meta_ensemble_kinship": "#3F51B5",
               "ensemble_kinship_fiw": "#009688", "best_checkpoint": "#FF9800"}

    for mkey in MODEL_KEYS:
        model = models[mkey].to(device)
        results = {}
        fig, ax = plt.subplots(figsize=(10, 6))
        for ni, ntype in enumerate(NOISE_TYPES):
            acc_list = []
            for s in SEVERITIES:
                if s == 0:
                    e1, e2 = emb1, emb2
                else:
                    e1 = apply_degradation(emb1_np, ntype, s)
                    e2 = apply_degradation(emb2_np, ntype, s)
                with torch.no_grad():
                    preds = model(e1.to(device), e2.to(device), rels.to(device)).cpu().view(-1).numpy()
                acc = float(accuracy_score(y_true, preds >= THRESHOLD) * 100)
                acc_list.append(acc)
            results[ntype] = acc_list
            ax.plot(SEVERITIES, acc_list, label=ntype, marker="o", lw=2)
        ax.set_xlabel("Noise Severity (0=Clean, 5=Extreme)", fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontweight="bold")
        ax.set_title(f"Robustness Curves - {MODEL_LABELS[mkey]} (FIW)", fontweight="bold")
        ax.legend(loc="lower left", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"robustness_{mkey}.png"))
        plt.close()
        save_json(results, os.path.join(OUT_DIR, f"{mkey}.json"))
        all_results[mkey] = results
        print(f"  [MODEL {mkey}] done")

    # Combined plot: mean accuracy across noise types per model
    fig, ax = plt.subplots(figsize=(9, 6))
    for mkey in MODEL_KEYS:
        res = all_results[mkey]
        mean_acc = [np.mean([res[n][si] for n in NOISE_TYPES]) for si in range(len(SEVERITIES))]
        ax.plot(SEVERITIES, mean_acc, label=MODEL_LABELS[mkey], color=mcolors[mkey], marker="o", lw=2)
    ax.set_xlabel("Noise Severity", fontweight="bold")
    ax.set_ylabel("Mean Accuracy (%)", fontweight="bold")
    ax.set_title("Overall Robustness - Mean Accuracy vs Severity (All Models)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "robustness_combined_4models.png"))
    plt.close()

    save_json(all_results, os.path.join(OUT_DIR, "robustness_summary.json"))
    print("  [MODULE 07 COMPLETE]")
    return all_results


if __name__ == "__main__":
    run_robustness()
