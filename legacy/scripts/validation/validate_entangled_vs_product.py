#!/usr/bin/env python3
"""
Validation script for entangled encoding vs product-state encoding.
Evaluates the entangled encoding on validation set
and compares against product-state (classical) encoding.
"""

import os
import sys
import json
import torch
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# Add project root to path - calculate correctly based on this file's location
current_file = os.path.abspath(__file__)  # .../scripts/validation/validate_entangled_vs_product.py
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # Go up three levels to get project root
sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier
from src.data_loaders import load_kinfacew_pairs, load_tskinface_pairs, prepare_pair_tensors
from scripts.evaluation.test_ensemble_on_unseen import load_fiw_pairs

def load_validation_data():
    """Load validation datasets using the same approach as ablation study."""

    datasets = {}

    # Load KinFaceW-I - use kinfacew-i cache
    try:
        k1_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-I"))
        k1_cache_path = os.path.join(project_root, "weights", "caches", "kinfacew-i_emb_cache.pkl")
        if os.path.exists(k1_cache_path):
            with open(k1_cache_path, "rb") as f:
                k1_cache = pickle.load(f)
            k1_t = prepare_pair_tensors(k1_pairs, k1_cache)
            print(f"Loaded KinFaceW-I: {len(k1_pairs)} pairs")
        else:
            print(f"Warning: KinFaceW-I cache not found at {k1_cache_path}")
            k1_t = None
    except Exception as e:
        print(f"Warning: Could not load KinFaceW-I: {e}")
        k1_t = None

    # Load KinFaceW-II - use general embeddings cache
    try:
        k2_pairs = load_kinfacew_pairs(os.path.join(project_root, "KinFaceW-II"))
        k2_cache_path = os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl")
        if os.path.exists(k2_cache_path):
            with open(k2_cache_path, "rb") as f:
                k2_cache = pickle.load(f)
            k2_t = prepare_pair_tensors(k2_pairs, k2_cache)
            print(f"Loaded KinFaceW-II: {len(k2_pairs)} pairs")
        else:
            print(f"Warning: General embeddings cache not found at {k2_cache_path}")
            k2_t = None
    except Exception as e:
        print(f"Warning: Could not load KinFaceW-II: {e}")
        k2_t = None

    # Load TSKinFace - use general embeddings cache
    try:
        ts_pairs = load_tskinface_pairs(os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped"))
        ts_cache_path = os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl")
        if os.path.exists(ts_cache_path):
            with open(ts_cache_path, "rb") as f:
                ts_cache = pickle.load(f)
            ts_t = prepare_pair_tensors(ts_pairs, ts_cache)
            print(f"Loaded TSKinFace: {len(ts_pairs)} pairs")
        else:
            print(f"Warning: General embeddings cache not found at {ts_cache_path}")
            ts_t = None
    except Exception as e:
        print(f"Warning: Could not load TSKinFace: {e}")
        ts_t = None

    # Load FIW - use fiw cache
    try:
        fiw_pairs = load_fiw_pairs(os.path.join(project_root, "public"), max_pairs=500)
        fiw_cache_paths = [
            os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl"),
            os.path.join(project_root, "outputs", "fiw_retraining_improved", "fiw_improved_cache.pkl"),
            os.path.join(project_root, "weights", "caches", "embeddings_cache.pkl"),
        ]
        fiw_cache = {}
        for p in fiw_cache_paths:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        fiw_cache = pickle.load(f)
                    print(f"Loaded FIW cache from {p}")
                    break
                except Exception:
                    pass

        if fiw_cache:
            fiw_t = prepare_pair_tensors(fiw_pairs, fiw_cache)
            print(f"Loaded FIW: {len(fiw_pairs)} pairs")
        else:
            print(f"Warning: FIW cache not found")
            fiw_t = None
    except Exception as e:
        print(f"Warning: Could not load FIW: {e}")
        fiw_t = None

    # Add datasets to dictionary
    if k1_t is not None:
        datasets["KinFaceW-I"] = k1_t
    if k2_t is not None:
        datasets["KinFaceW-II"] = k2_t
    if ts_t is not None:
        datasets["TSKinFace"] = ts_t
    if fiw_t is not None:
        datasets["FIW"] = fiw_t

    return datasets

def evaluate_model(model, datasets, threshold=0.5):
    """Evaluate model on given datasets."""
    results = {}
    model.eval()

    for d_name, d_tensors in datasets.items():
        emb1, emb2, y_true, rels = d_tensors

        # Debug: print tensor shapes
        print(f"{d_name} - emb1 shape: {emb1.shape}, emb2 shape: {emb2.shape}, y_true shape: {y_true.shape}, rels shape: {rels.shape}")

        with torch.no_grad():
            # Handle device placement
            device = next(model.parameters()).device
            emb1 = emb1.to(device)
            emb2 = emb2.to(device)
            rels = rels.to(device)

            predictions = model(emb1, emb2, rels).cpu().view(-1).numpy()

        y_true_np = y_true.view(-1).numpy()
        binary_predictions = (predictions >= threshold).astype(float)
        accuracy = accuracy_score(y_true_np, binary_predictions) * 100

        results[d_name] = {
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'binary_predictions': binary_predictions.tolist()
        }

        print(f"{d_name}: {accuracy:.2f}% accuracy")

    return results

def main():
    print("=" * 60)
    print("ENTANGLED ENCODING VS PRODUCT-STATE ENCODING VALIDATION")
    print("=" * 60)

    # Load validation data
    print("\nLoading validation datasets...")
    datasets = load_validation_data()

    if not datasets:
        print("ERROR: No validation datasets loaded!")
        return 1

    # Test Entangled Model (QuantumInspiredCrossAttention + entangled encoding)
    print("\n" + "-" * 50)
    print("Testing ENTANGLED Encoding (QuantumInspiredCrossAttention)")
    print("-" * 50)

    model_entangled = HybridKinshipClassifier(
        n_qubits=8,
        encoding_mode="entangled",
        projection_type="quantum_inspired_attention"
    )
    model_entangled.eval()

    entangled_results = evaluate_model(model_entangled, datasets)

    # Test Product-State Model (QuantumInspiredCrossAttention + product encoding)
    print("\n" + "-" * 50)
    print("Testing PRODUCT-STATE Encoding (QuantumInspiredCrossAttention)")
    print("-" * 50)

    model_product = HybridKinshipClassifier(
        n_qubits=8,
        encoding_mode="product",
        projection_type="quantum_inspired_attention"
    )
    model_product.eval()

    product_results = evaluate_model(model_product, datasets)

    # Calculate improvements (entangled over product)
    print("\n" + "=" * 50)
    print("IMPROVEMENT SUMMARY (Entangled over Product)")
    print("=" * 50)

    improvements = {}
    for dataset in datasets.keys():
        if dataset in entangled_results and dataset in product_results:
            entangled_acc = entangled_results[dataset]['accuracy']
            product_acc = product_results[dataset]['accuracy']
            improvement = entangled_acc - product_acc
            improvements[dataset] = improvement

            print(f"{dataset}:")
            print(f"  Entangled: {entangled_acc:.2f}%")
            print(f"  Product:   {product_acc:.2f}%")
            print(f"  Improvement: {improvement:+.2f}%")
            print()

    # Overall assessment
    avg_improvement = np.mean(list(improvements.values())) if improvements else 0
    print(f"Average improvement across datasets: {avg_improvement:+.2f}%")

    # Save results
    output_dir = os.path.join(project_root, "outputs", "validation")
    os.makedirs(output_dir, exist_ok=True)

    results_data = {
        'entangled_model': entangled_results,
        'product_model': product_results,
        'improvements': improvements,
        'average_improvement': float(avg_improvement)
    }

    output_path = os.path.join(output_dir, "entangled_vs_product_validation.json")
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nValidation results saved to: {output_path}")

    # Determine if entangled encoding is beneficial
    if avg_improvement > 0:
        print("��✓ Entangled encoding shows positive improvement over product-state encoding!")
        return 0
    else:
        print("��⚠ Entangled encoding does not show improvement over product-state encoding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())