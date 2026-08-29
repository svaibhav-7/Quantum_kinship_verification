# -*- coding: utf-8 -*-
"""
MODULE 7: EMBEDDING-SPACE PERTURBATION SENSITIVITY ANALYSIS
Evaluates sensitivity of learned features to various types of perturbations in embedding space.
NOTE: This is not equivalent to real-world image robustness testing.
For true robustness evaluation, perturbations should be applied to input images before feature extraction.
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
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(current_dir)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier
from src.data_loaders import prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

def apply_embedding_perturbation(emb_tensor, noise_type, severity):
    """
    Applies perturbations to 512D feature vectors for sensitivity analysis.
    NOTE: These are not equivalent to real-world image degradations.
    """
    emb_np = emb_tensor.numpy()
    np.random.seed(42 + severity)

    if noise_type == "Gaussian Noise":
        # Additive Gaussian noise
        noise = np.random.normal(0, 0.03 * severity, emb_np.shape)
        degraded = emb_np + noise
    elif noise_type == "Gaussian Blur":
        # 1D Gaussian smoothing across embedding dimensions
        kernel_size = max(3, 2 * severity + 1)
        sigma = 0.5 * severity + 0.1
        k = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / (sigma ** 2))
        k = k / k.sum()
        degraded = np.zeros_like(emb_np)
        for i in range(emb_np.shape[0]):
            degraded[i] = np.convolve(emb_np[i], k, mode='same')
    elif noise_type == "JPEG Compression":
        # Quantization-like effect
        factor = 10.0 / (severity + 1e-5)
        degraded = np.round(emb_np * factor) / factor
    elif noise_type == "Random Occlusion":
        # Zero-out random dimensions
        mask = np.random.binomial(1, 1.0 - 0.06 * severity, emb_np.shape)
        degraded = emb_np * mask
    elif noise_type == "Brightness Shifting":
        # Uniform shift
        degraded = emb_np + 0.02 * severity
    elif noise_type == "Rotation / Tilt":
        # Random permutation of dimensions (questionable as isomorphism)
        perm = np.random.permutation(emb_np.shape[1])
        weight = 0.04 * severity
        degraded = (1.0 - weight) * emb_np + weight * emb_np[:, perm]
    else:
        degraded = emb_np

    # Re-normalize L2 (common practice for face embeddings)
    norms = np.linalg.norm(degraded, axis=1, keepdims=True) + 1e-8
    return torch.tensor(degraded / norms, dtype=torch.float32)

def run_robustness_tests():
    print("\n" + "="*70)
    print("  MODULE 7: EMBEDDING-SPACE PERTURBATION SENSITIVITY ANALYSIS")
    print("="*70)

    out_dir = os.path.join(research_root, "outputs", "07_robustness_degradation")
    os.makedirs(out_dir, exist_ok=True)

    fiw_cache_path = os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl")
    try:
        with open(fiw_cache_path, "rb") as f:
            fiw_cache = pickle.load(f)
        norm_fiw = {os.path.normcase(os.path.abspath(k)): v for k, v in fiw_cache.items()}
        emb1, emb2, y_true_t, rels = prepare_pair_tensors(load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500), norm_fiw)
        y_true = y_true_t.view(-1).numpy()
        print(f"  Loaded FIW evaluation set: {len(y_true)} pairs")
    except Exception as e:
        print(f"  [ERROR] Failed to load FIW dataset: {e}")
        return {"status": "ERROR", "message": f"Failed to load dataset: {e}"}

    meta_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    try:
        m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
        m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
        m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")

        full_meta = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
        full_meta.load_state_dict(torch.load(meta_path, map_location="cpu"))
        full_meta.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        full_meta.to(device)
        print(f"  Model loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return {"status": "ERROR", "message": f"Failed to load model: {e}"}

    noise_types = ["Gaussian Noise", "Gaussian Blur", "JPEG Compression", "Random Occlusion", "Brightness Shifting", "Rotation / Tilt"]
    severities = [0, 1, 2, 3, 4, 5]

    robustness_results = {}

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E91E63", "#3F51B5", "#009688", "#FF9800", "#9C27B0", "#795548"]

    for n_idx, n_type in enumerate(noise_types):
        acc_list = []
        for s in severities:
            if s == 0:
                e1_deg, e2_deg = emb1, emb2
            else:
                e1_deg = apply_embedding_perturbation(emb1, n_type, s)
                e2_deg = apply_embedding_perturbation(emb2, n_type, s)

            with torch.no_grad():
                preds = full_meta(e1_deg.to(device), e2_deg.to(device), rels.to(device)).cpu().view(-1).numpy()
            acc = float(accuracy_score(y_true, preds >= 0.5216) * 100)
            acc_list.append(acc)

        robustness_results[n_type] = acc_list
        ax.plot(severities, acc_list, label=n_type, color=colors[n_idx], marker="o", lw=2)

    ax.set_xlabel("Perturbation Severity Level (0 = None, 5 = Strong)", fontweight="bold")
    ax.set_ylabel("Classification Accuracy (%)", fontweight="bold")
    ax.set_title("Embedding-Space Perturbation Sensitivity Analysis\n(NOTE: Not equivalent to real-world image robustness)", fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add honesty disclaimer as text on plot
    ax.text(0.02, 0.98, "IMPORTANT: This analysis applies perturbations\nto 512D feature vectors, not to input images.\nReal-world robustness requires testing\non perturbed input images.",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    p1 = os.path.join(out_dir, "robustness_degradation_curves.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    save_path = os.path.join(out_dir, "robustness_results.json")
    with open(save_path, "w") as f:
        json.dump(robustness_results, f, indent=2)

    print(f"Generated Perturbation Sensitivity Plot: {p1}")
    print("\nIMPORTANT LIMITATIONS OF THIS ANALYSIS:")
    print("- Perturbations applied to 512D embeddings, not input images")
    print("- 'Gaussian Blur' and 'Rotation/Tilt' have questionable interpretations in embedding space")
    print("- For true robustness evaluation, apply perturbations to images before feature extraction")
    print("- Results indicate sensitivity of learned feature representation to various noise types")
    print(f"[MODULE 7 COMPLETE] Saved to {save_path}")
    return robustness_results

if __name__ == "__main__":
    run_robustness_tests()