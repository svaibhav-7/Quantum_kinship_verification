# -*- coding: utf-8 -*-
"""
COMMON METRICS & PLOTTING UTILITY for Master Model Comparison.

Provides shared functions for per-model metric computation, dataset coloring,
histogram/curve plotting, and JSON serialization helpers used across all 12 modules.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

DATASET_COLORS = {
    "KinFaceW-I": "#1976D2",
    "KinFaceW-II": "#009688",
    "TSKinFace": "#FF9800",
    "FIW": "#9C27B0",
}

MODEL_COLORS = {
    "ensemble_kinship_full": "#E91E63",
    "meta_ensemble_kinship": "#3F51B5",
    "ensemble_kinship_fiw": "#009688",
    "best_checkpoint": "#FF9800",
}

MODEL_SHORT = {
    "ensemble_kinship_full": "Base Ens",
    "meta_ensemble_kinship": "Meta-Ens",
    "ensemble_kinship_fiw": "FIW Ens",
    "best_checkpoint": "Best Ckpt",
}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_convert)
    return path


def _convert(o):
    if isinstance(o, (np.ndarray, np.generic)):
        return o.tolist()
    return str(o)


def compute_dataset_metrics(y_true, preds, threshold=0.5):
    """Compute a full set of classification metrics for one model on one dataset."""
    y_true = np.asarray(y_true).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    binary = (preds >= threshold).astype(int)

    fpr, tpr, thr = roc_curve(y_true, preds)
    roc_auc = float(auc(fpr, tpr))
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, preds)
    pr_auc = float(average_precision_score(y_true, preds))
    acc = float(accuracy_score(y_true, binary) * 100)
    f1 = float(f1_score(y_true, binary, zero_division=0) * 100)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, binary).ravel()
    except ValueError:
        tn = tp = fp = fn = 0

    return {
        "n_samples": int(len(y_true)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1_score": f1,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": pr_prec.tolist(),
        "recall": pr_rec.tolist(),
        "preds": preds.tolist(),
        "y_true": y_true.tolist(),
        "kin_mean": float(np.mean(preds[y_true == 1]) if np.any(y_true == 1) else 0),
        "nonkin_mean": float(np.mean(preds[y_true == 0]) if np.any(y_true == 0) else 0),
    }


def per_model_summary(module_dir, model_key, name, data):
    """Write a per-model summary JSON into a module subfolder."""
    path = os.path.join(module_dir, f"{model_key}.json")
    payload = {"model": model_key, "module": name, **data}
    save_json(payload, path)
    return path

