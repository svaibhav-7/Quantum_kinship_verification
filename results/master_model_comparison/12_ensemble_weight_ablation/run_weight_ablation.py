# -*- coding: utf-8 -*-
"""
MODULE 12: ENSEMBLE FUSION WEIGHT ABLATION (per model)

For the meta-ensemble model, sweeps the domain fusion weights and produces
per-config metrics. Also records single-model / ensemble configurations for
the other 3 models as baselines. Produces per-config JSON + a plot.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
_project_root = os.path.dirname(os.path.dirname(_module_root))
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_data import get_datasets
from common_utils import save_json

OUT_DIR = os.path.join(_module_root, "12_ensemble_weight_ablation")
os.makedirs(OUT_DIR, exist_ok=True)
THRESHOLD = 0.5

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier


def run_weight_ablation():
    print("\n" + "=" * 70)
    print("  MODULE 12: ENSEMBLE FUSION WEIGHT ABLATION (4 MODELS)")
    print("=" * 70)

    datasets = get_datasets()
    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb1, emb2, y_t, rels, _ = datasets["FIW"]
    y_true = y_t.view(-1).numpy()

    # Weight configurations for the meta-ensemble
    configs = {
        "Equal Weighting (0.33,0.33,0.33)": (0.333, 0.333, 0.334),
        "Base-Dominant (0.70,0.20,0.10)": (0.70, 0.20, 0.10),
        "FIW-Dominant (0.20,0.70,0.10)": (0.20, 0.70, 0.10),
        "Learned Optimal (0.45,0.35,0.20)": (0.45, 0.35, 0.20),
    }

    # Build meta-ensemble architecture and reload weights for weight abaltion
    m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")

    all_results = {}

    # Individual model results (as baselines)
    for mkey in MODEL_KEYS:
        model = models[mkey].to(device)
        with torch.no_grad():
            preds = model(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
        all_results[MODEL_LABELS[mkey]] = _compute(preds, y_true)
        save_json(all_results[MODEL_LABELS[mkey]],
                  os.path.join(OUT_DIR, f"{mkey}_baseline.json"))
        print(f"  [MODEL {mkey} baseline] acc={all_results[MODEL_LABELS[mkey]]['accuracy']:.2f}%")

    # Weight ablation for meta-ensemble
    meta_state = torch.load(
        os.path.join(_project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt"),
        map_location="cpu", weights_only=True)

    for cname, weights in configs.items():
        meta_model = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=weights)
        meta_model.load_state_dict(meta_state)
        meta_model.eval().to(device)
        with torch.no_grad():
            preds = meta_model(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
        res = _compute(preds, y_true)
        res["weights"] = list(weights)
        all_results[cname] = res
        save_json(res, os.path.join(OUT_DIR, f"config_{cname}.json".replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_").replace(".", "p")))
        print(f"  [CONFIG {cname}] acc={res['accuracy']:.2f}%, auc={res['roc_auc']:.4f}")

    _plot_combined(all_results)
    save_json(all_results, os.path.join(OUT_DIR, "weight_ablation_summary.json"))
    print("  [MODULE 12 COMPLETE]")
    return all_results


def _compute(preds, y_true):
    binary = (preds >= THRESHOLD).astype(int)
    auc = roc_auc_score(y_true, preds) if len(np.unique(y_true)) == 2 else 0.5
    return {
        "accuracy": float(accuracy_score(y_true, binary) * 100),
        "roc_auc": float(auc),
        "f1_score": float(f1_score(y_true, binary, zero_division=0) * 100),
    }


def _plot_combined(all_results):
    names = list(all_results.keys())
    accs = [all_results[n]["accuracy"] for n in names]
    aucs = [all_results[n]["roc_auc"] * 100 for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, accs, w, label="Accuracy (%)", color="#009688")
    ax.bar(x + w / 2, aucs, w, label="ROC-AUC (%)", color="#9C27B0")
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax.text(i - w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, u + 0.5, f"{u:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Ensemble Fusion Weight Ablation & Model Baselines (FIW)", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "weight_ablation_combined.png"))
    plt.close()


if __name__ == "__main__":
    run_weight_ablation()
