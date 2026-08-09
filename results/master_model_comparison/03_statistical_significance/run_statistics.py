# -*- coding: utf-8 -*-
"""
MODULE 03: STATISTICAL SIGNIFICANCE (per model)

For each of the 4 models, computes accuracy, 95% bootstrap CI, and a paired
comparison vs. the best single model (best_checkpoint) using McNemar test and
paired t-test on the FIW dataset. Produces per-model JSON + a significance plot.
"""

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets
from common_utils import save_json

OUT_DIR = os.path.join(_module_root, "03_statistical_significance")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.5


def bootstrap_ci(y_true, preds, n_bootstraps=1000, ci=95):
    rng = np.random.default_rng(42)
    n = len(y_true)
    accs = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        accs.append(accuracy_score(y_true[idx], preds[idx] >= THRESHOLD) * 100)
    return float(np.percentile(accs, (100 - ci) / 2)), float(np.percentile(accs, 100 - (100 - ci) / 2))


def mcnemar(y_true, pred_a, pred_b, threshold=THRESHOLD):
    ca = (pred_a >= threshold) == y_true
    cb = (pred_b >= threshold) == y_true
    b = np.sum(ca & ~cb)
    c = np.sum(~ca & cb)
    if b + c == 0:
        return 0.0, 1.0
    chi2 = float((abs(b - c) - 1) ** 2 / (b + c))
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return chi2, p


def run_statistics():
    print("\n" + "=" * 70)
    print("  MODULE 03: STATISTICAL SIGNIFICANCE (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    # Use FIW dataset for significance testing
    dname = "FIW"
    emb1, emb2, y_true_t, rels, _ = datasets[dname]
    y_true = y_true_t.view(-1).numpy()

    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_dict = {}
    for mkey in MODEL_KEYS:
        m = models[mkey].to(device)
        with torch.no_grad():
            preds_dict[mkey] = m(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()

    results = {}
    for mkey in MODEL_KEYS:
        preds = preds_dict[mkey]
        acc = float(accuracy_score(y_true, preds >= THRESHOLD) * 100)
        ci_low, ci_high = bootstrap_ci(y_true, preds)

        # Compare against best_checkpoint (single model baseline)
        base_preds = preds_dict["best_checkpoint"]
        chi2, p_mcnemar = mcnemar(y_true, preds, base_preds)
        err_m = np.abs(y_true - preds)
        err_b = np.abs(y_true - base_preds)
        t_stat, t_p = stats.ttest_rel(err_m, err_b) if np.std(err_m - err_b) > 0 else (0.0, 1.0)

        results[mkey] = {
            "model": MODEL_LABELS[mkey],
            "dataset": dname,
            "sample_count": int(len(y_true)),
            "accuracy": acc,
            "bootstrap_95_ci": [ci_low, ci_high],
            "mcnemar_chi2": chi2,
            "mcnemar_p_value": p_mcnemar,
            "paired_ttest_stat": float(t_stat),
            "paired_ttest_p_value": float(t_p),
            "significant_vs_best_ckpt": bool(p_mcnemar < 0.05),
        }
        save_json(results[mkey], os.path.join(OUT_DIR, f"{mkey}.json"))
        print(f"  [MODEL {mkey}] acc={acc:.2f}%, CI=[{ci_low:.2f},{ci_high:.2f}], McNemar p={p_mcnemar:.4f}")

    # Plot comparison bar chart
    names = [MODEL_LABELS[k] for k in MODEL_KEYS]
    accs = [results[k]["accuracy"] for k in MODEL_KEYS]
    ci_lows = [results[k]["bootstrap_95_ci"][0] for k in MODEL_KEYS]
    ci_highs = [results[k]["bootstrap_95_ci"][1] for k in MODEL_KEYS]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bar = ax.bar(x, accs, 0.5, color=["#E91E63", "#3F51B5", "#009688", "#FF9800"], alpha=0.85)
    err = [np.array([accs[i] - ci_lows[i], ci_highs[i] - accs[i]]).reshape(2, 1) for i in range(len(accs))]
    ax.errorbar(x, accs, yerr=np.array([[accs[i] - ci_lows[i], ci_highs[i] - accs[i]] for i in range(len(accs))]).T, fmt="none", ecolor="black", capsize=5)
    for i, r in enumerate(bar):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.5, f"{accs[i]:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("FIW Accuracy (%)", fontweight="bold")
    ax.set_title("Statistical Significance - Accuracy with 95% Bootstrap CI (FIW)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "statistical_significance_bar_chart.png"))
    plt.close()

    save_json(results, os.path.join(OUT_DIR, "statistical_summary.json"))
    print("  [MODULE 03 COMPLETE]")
    return results


if __name__ == "__main__":
    run_statistics()
