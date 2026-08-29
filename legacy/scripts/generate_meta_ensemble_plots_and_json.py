# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- META-ENSEMBLE COMPREHENSIVE METRICS & PLOTS
=================================================================================

Generates detailed JSON files and publication-quality plots comparing:
  1. Model 1: ensemble_kinship_full.pt (Base Ensemble)
  2. Model 2: ensemble_kinship_fiw.pt (FIW 5-Fold Ensemble)
  3. Model 3: best_checkpoint.pt (Newly Fine-Tuned Model)
  4. Meta-Ensemble: meta_ensemble_kinship.pt (Domain-Weighted Meta Ensemble)
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "evaluation", "training", "inference"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import load_kinfacew_pairs, prepare_pair_tensors

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "Model 1 (Base Ensemble)": "#2196F3",
    "Model 2 (FIW Ensemble)": "#FF9800",
    "Model 3 (Fine-Tuned Checkpoint)": "#E91E63",
    "Meta-Ensemble (Saved Model)": "#9C27B0",
}

def load_all_models():
    # Model 1
    m1_path = os.path.join(project_root, "weights", "active_ensemble", "ensemble_kinship_full.pt")
    models1 = [HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)]
    ens1 = EnsembleKinshipClassifier(models1)
    ens1.load_state_dict(torch.load(m1_path, map_location="cpu"))
    ens1.eval()

    # Model 2
    m2_path = os.path.join(project_root, "weights", "secondary_active_ensemble", "ensemble_kinship_fiw.pt")
    models2 = [HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)]
    ens2 = EnsembleKinshipClassifier(models2)
    ens2.load_state_dict(torch.load(m2_path, map_location="cpu"))
    ens2.eval()

    # Model 3
    m3_path = os.path.join(project_root, "outputs", "fiw_retraining_improved", "best_checkpoint.pt")
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    m3.load_state_dict(torch.load(m3_path, map_location="cpu"))
    m3.eval()

    # Meta-Ensemble
    ens1_meta = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    ens2_meta = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3_meta = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    meta_m = MetaEnsembleKinshipClassifier(ens1_meta, ens2_meta, m3_meta, weights=(0.45, 0.35, 0.20))
    meta_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    meta_m.load_state_dict(torch.load(meta_path, map_location="cpu"))
    meta_m.eval()

    return {
        "Model 1 (Base Ensemble)": ens1,
        "Model 2 (FIW Ensemble)": ens2,
        "Model 3 (Fine-Tuned Checkpoint)": m3,
        "Meta-Ensemble (Saved Model)": meta_m,
    }

def predict_batched(model, emb1, emb2, rels, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preds = []
    n = len(emb1)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            p = model(emb1[i:i+batch_size].to(device), emb2[i:i+batch_size].to(device), rels[i:i+batch_size].to(device)).cpu().numpy().flatten()
            preds.extend(p)
    return np.array(preds)

def full_eval(y_true, preds, rels_tensor):
    fpr, tpr, thresholds = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_thresh = float(thresholds[opt_idx])
    
    binary_opt = (preds >= opt_thresh).astype(int)
    binary_05 = (preds >= 0.50).astype(int)
    
    acc_opt = accuracy_score(y_true, binary_opt)
    acc_05 = accuracy_score(y_true, binary_05)
    prec_opt, rec_opt, f1_opt, _ = precision_recall_fscore_support(y_true, binary_opt, average="binary", zero_division=0)
    prec_05, rec_05, f1_05, _ = precision_recall_fscore_support(y_true, binary_05, average="binary", zero_division=0)
    
    cm_opt = confusion_matrix(y_true, binary_opt).tolist()
    cm_05 = confusion_matrix(y_true, binary_05).tolist()
    
    rel_indices = torch.argmax(rels_tensor, dim=1).cpu().numpy()
    rel_names = {0: "Father-Daughter (FD)", 1: "Father-Son (FS)", 2: "Mother-Daughter (MD)", 3: "Mother-Son (MS)"}
    per_rel = {}
    for r_idx, r_name in rel_names.items():
        mask = (rel_indices == r_idx)
        if np.sum(mask) > 0:
            rel_acc = float(accuracy_score(y_true[mask], binary_opt[mask]) * 100)
            rel_count = int(np.sum(mask))
            per_rel[r_name] = {"accuracy": rel_acc, "sample_count": rel_count}
            
    return {
        "n_samples": len(y_true),
        "roc_auc": float(roc_auc),
        "optimal_threshold": opt_thresh,
        "optimal_accuracy": float(acc_opt * 100),
        "optimal_precision": float(prec_opt * 100),
        "optimal_recall": float(rec_opt * 100),
        "optimal_f1_score": float(f1_opt * 100),
        "default_05_accuracy": float(acc_05 * 100),
        "default_05_precision": float(prec_05 * 100),
        "default_05_recall": float(rec_05 * 100),
        "default_05_f1_score": float(f1_05 * 100),
        "confusion_matrix_optimal": cm_opt,
        "confusion_matrix_default": cm_05,
        "per_relation_breakdown": per_rel,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "preds": preds.tolist(),
    }

def main():
    print("=================================================================")
    print("GENERATING META-ENSEMBLE METRICS JSON & PUBLICATION PLOTS")
    print("=================================================================")
    
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    models = load_all_models()
    
    print("\nLoading KinFaceW-I benchmark data (1,066 pairs)...")
    kin_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-I"))
    with open(os.path.join(project_root, "weights", "caches", "kinfacew-i_emb_cache.pkl"), "rb") as f:
        kin_c = pickle.load(f)
    norm_kin_c = {os.path.normcase(os.path.abspath(k)): v for k, v in kin_c.items()}
    kin_t = prepare_pair_tensors(kin_pairs, norm_kin_c)
    y_kin = kin_t[2].view(-1).numpy()

    eval_data = {}
    print("\nComputing model predictions...")
    for name, m in models.items():
        print(f"  Evaluating {name}...")
        p_kin = predict_batched(m, kin_t[0], kin_t[1], kin_t[3])
        eval_data[name] = full_eval(y_kin, p_kin, kin_t[3])

    # Save JSON files
    json_path = os.path.join(results_dir, "meta_ensemble_comprehensive_metrics.json")
    json_clean = {}
    for m_name, metrics in eval_data.items():
        m_copy = {k: v for k, v in metrics.items() if k not in ["fpr", "tpr", "preds"]}
        json_clean[m_name] = m_copy
            
    with open(json_path, "w") as f:
        json.dump(json_clean, f, indent=2)
    print(f"\n[SAVED JSON] {json_path}")

    # PLOT 1: ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for m_name, metrics in eval_data.items():
        ax.plot(metrics["fpr"], metrics["tpr"], label=f"{m_name} (AUC = {metrics['roc_auc']:.4f})", color=COLORS[m_name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="Random Chance (AUC = 0.50)")
    ax.set_title("ROC Curves Comparison on KinFaceW-I Benchmark")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_plot_path = os.path.join(plots_dir, "meta_ensemble_roc_curves.png")
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"[SAVED PLOT] {roc_plot_path}")

    # PLOT 2: Multi-Metric Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(eval_data.keys())
    x = np.arange(len(model_names))
    width = 0.18

    auc_vals = [eval_data[m]["roc_auc"] * 100 for m in model_names]
    acc_vals = [eval_data[m]["optimal_accuracy"] for m in model_names]
    prec_vals = [eval_data[m]["optimal_precision"] for m in model_names]
    rec_vals = [eval_data[m]["optimal_recall"] for m in model_names]

    ax.bar(x - 1.5*width, auc_vals, width, label="ROC-AUC (%)", color="#3F51B5")
    ax.bar(x - 0.5*width, acc_vals, width, label="Optimal Accuracy (%)", color="#4CAF50")
    ax.bar(x + 0.5*width, prec_vals, width, label="Precision (%)", color="#FF9800")
    ax.bar(x + 1.5*width, rec_vals, width, label="Recall (%)", color="#E91E63")

    ax.set_title("Comprehensive Performance Comparison (KinFaceW-I Benchmark)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("(")[0].strip() for m in model_names], rotation=15, ha="right")
    ax.set_ylim(40, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    metrics_plot_path = os.path.join(plots_dir, "meta_ensemble_metrics_comparison.png")
    plt.savefig(metrics_plot_path)
    plt.close()
    print(f"[SAVED PLOT] {metrics_plot_path}")

    # PLOT 3: Per-Relation Accuracy Breakdown Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    rel_names = ["Father-Daughter (FD)", "Father-Son (FS)", "Mother-Daughter (MD)", "Mother-Son (MS)"]
    x = np.arange(len(rel_names))
    width = 0.20

    for idx, (m_name, m_color) in enumerate(COLORS.items()):
        rel_dict = eval_data[m_name]["per_relation_breakdown"]
        accs = [rel_dict.get(r, {}).get("accuracy", 0.0) for r in rel_names]
        ax.bar(x + (idx - 1.5) * width, accs, width, label=m_name.split("(")[0].strip(), color=m_color)

    ax.set_title("Per-Relation Accuracy Breakdown across Kinship Types")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Father-Daughter\n(FD)", "Father-Son\n(FS)", "Mother-Daughter\n(MD)", "Mother-Son\n(MS)"])
    ax.set_ylim(40, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    rel_plot_path = os.path.join(plots_dir, "meta_ensemble_per_relation_accuracy.png")
    plt.savefig(rel_plot_path)
    plt.close()
    print(f"[SAVED PLOT] {rel_plot_path}")

    print("\n=================================================================")
    print("ALL JSON METRICS & PLOTS SUCCESSFULLY CREATED!")
    print("=================================================================")

if __name__ == "__main__":
    main()
