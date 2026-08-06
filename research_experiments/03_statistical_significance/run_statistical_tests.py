# -*- coding: utf-8 -*-
"""
MODULE 3: STATISTICAL SIGNIFICANCE ANALYSIS
Computes McNemar's Test, 95% Bootstrap Confidence Intervals, Paired t-tests, and p-values.
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import load_kinfacew_pairs, load_tskinface_pairs, prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

def bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95):
    np.random.seed(42)
    accs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx] >= 0.5216) * 100)
    lower = np.percentile(accs, (100 - ci) / 2)
    upper = np.percentile(accs, 100 - (100 - ci) / 2)
    return float(lower), float(upper)

def mcnemar_test(y_true, pred_a, pred_b, threshold=0.5216):
    correct_a = (pred_a >= threshold) == y_true
    correct_b = (pred_b >= threshold) == y_true
    
    b = np.sum(correct_a & ~correct_b) # A correct, B wrong
    c = np.sum(~correct_a & correct_b) # A wrong, B correct

    if (b + c) == 0:
        return 0.0, 1.0
    chi2 = float((abs(b - c) - 1)**2 / (b + c))
    p_val = float(1 - stats.chi2.cdf(chi2, df=1))
    return chi2, p_val

def run_statistical_tests():
    print("\n" + "="*70)
    print("  MODULE 3: STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "03_statistical_significance")
    os.makedirs(out_dir, exist_ok=True)

    # Load Data & Model
    meta_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    
    full_meta = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
    full_meta.load_state_dict(torch.load(meta_path, map_location="cpu"))
    full_meta.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_meta.to(device)

    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    with open(fiw_cache_path, "rb") as f:
        fiw_cache = pickle.load(f)
    norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}
    fiw_t = prepare_pair_tensors(load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500), norm_fiw)

    emb1, emb2, y_true_t, rels = fiw_t
    y_true = y_true_t.view(-1).numpy()

    with torch.no_grad():
        meta_preds = full_meta(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()
        single_preds = m3.to(device)(emb1.to(device), emb2.to(device), rels.to(device)).cpu().view(-1).numpy()

    # Bootstrap CIs
    ci_low, ci_high = bootstrap_ci(y_true, meta_preds)

    # McNemar Test against Single Model Baseline
    chi2, p_val = mcnemar_test(y_true, meta_preds, single_preds)

    # Paired t-test on prediction errors
    err_meta = np.abs(y_true - meta_preds)
    err_single = np.abs(y_true - single_preds)
    t_stat, t_pval = stats.ttest_rel(err_meta, err_single)

    results = {
        "dataset": "Families In the Wild (FIW)",
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, meta_preds >= 0.5216) * 100),
        "bootstrap_95_ci": [ci_low, ci_high],
        "mcnemar_chi2": float(chi2),
        "mcnemar_p_value": float(p_val),
        "paired_ttest_stat": float(t_stat),
        "paired_ttest_p_value": float(t_pval),
        "statistically_significant": float(p_val) < 0.05
    }

    save_path = os.path.join(out_dir, "statistical_tests.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Dataset                 : {results['dataset']}")
    print(f"Accuracy                : {results['accuracy']:.2f}%")
    print(f"95% Bootstrap CI        : [{ci_low:.2f}%, {ci_high:.2f}%]")
    print(f"McNemar Chi-Squared     : {chi2:.4f} (p-value = {p_val:.4e})")
    print(f"Paired t-test t-stat    : {t_stat:.4f} (p-value = {t_pval:.4e})")
    print(f"Statistically Significant: {'YES (p < 0.05)' if p_val < 0.05 else 'NO'}")

    print(f"\n[MODULE 3 COMPLETE] Saved to {save_path}")
    return results

if __name__ == "__main__":
    run_statistical_tests()
