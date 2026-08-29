# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- BUILD & SAVE META-ENSEMBLE MODEL
=================================================================================

Bundles the 3 core models into a single MetaEnsembleKinshipClassifier model:
  1. Base Ensemble (ensemble_kinship_full.pt - 5 folds)
  2. FIW Ensemble (ensemble_kinship_fiw.pt - 5 folds)
  3. Fine-tuned Checkpoint (best_checkpoint.pt - 1 model)

Saves the combined PyTorch checkpoint to:
  `weights/active_ensemble/meta_ensemble_kinship.pt`
and metadata to:
  `weights/active_ensemble/meta_ensemble_metadata.json`

Usage:
  python scripts/training/build_meta_ensemble.py
"""

import os
import sys
import json
import torch

# Project root setup (supports .py, Jupyter, Google Colab, and Kaggle)
def setup_project_environment():
    try:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        start_dir = os.getcwd()

    curr = os.path.abspath(start_dir)
    while curr and curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "src", "models_improved.py")):
            if curr not in sys.path:
                sys.path.insert(0, curr)
            try:
                os.chdir(curr)
            except Exception:
                pass
            return curr
        curr = os.path.dirname(curr)

    try:
        for item in os.listdir(start_dir):
            full_path = os.path.join(start_dir, item)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "src", "models_improved.py")):
                if full_path not in sys.path:
                    sys.path.insert(0, full_path)
                try:
                    os.chdir(full_path)
                except Exception:
                    pass
                return full_path
    except Exception:
        pass

    if start_dir not in sys.path:
        sys.path.insert(0, start_dir)
    return start_dir

project_root = setup_project_environment()

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier

def load_base_ensemble(project_root):
    path = os.path.join(project_root, "weights", "active_ensemble", "ensemble_kinship_full.pt")
    models = [HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)]
    ens = EnsembleKinshipClassifier(models)
    state = torch.load(path, map_location="cpu")
    ens.load_state_dict(state)
    return ens

def load_fiw_ensemble(project_root):
    path = os.path.join(project_root, "weights", "active_ensemble", "ensemble_kinship_fiw.pt")
    models = [HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)]
    ens = EnsembleKinshipClassifier(models)
    state = torch.load(path, map_location="cpu")
    ens.load_state_dict(state)
    return ens

def load_fine_tuned_single(project_root):
    path = os.path.join(project_root, "outputs", "fiw_retraining_improved", "best_checkpoint.pt")
    model = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model




def get_model_accuracies():
    """Return hardcoded FIW accuracies for the three models (from IMPROVEMENT_SUMMARY.md)."""
    # These are the FIW accuracies for the three models:
    # Base Ensemble (ensemble_kinship_full.pt): 57.63%
    # FIW Ensemble (ensemble_kinship_fiw.pt): 65.62%
    # Fine-tuned Checkpoint (best_checkpoint.pt): 88.12%
    return (57.63, 65.62, 88.12)

def main():
    print("=================================================================")
    print("Building Quantum Kinship Meta-Ensemble Classifier...")
    print("=================================================================")

    print("[1/3] Loading Base Ensemble (5 folds)...")
    base_ens = load_base_ensemble(project_root)

    print("[2/3] Loading FIW Ensemble (5 folds)...")
    fiw_ens = load_fiw_ensemble(project_root)

    print("[3/3] Loading Fine-tuned Single Checkpoint...")
    single_fiw = load_fine_tuned_single(project_root)

    # Get model accuracies to inform weighting
    accuracies = get_model_accuracies()

    if accuracies is not None:
        # Calculate performance-based weights (proportional to accuracy)
        acc1, acc2, acc3 = accuracies
        total_acc = acc1 + acc2 + acc3
        weights = (acc1 / total_acc, acc2 / total_acc, acc3 / total_acc)
        print(f"  Using performance-based weights: {weights}")
    else:
        # Fallback to equal weighting if we can't get accuracies
        weights = (0.333, 0.333, 0.334)
        print(f"  Using equal weights: {weights}")

    # Construct Meta-Ensemble with performance-based or equal weighting
    meta_model = MetaEnsembleKinshipClassifier(
        ensemble_full=base_ens,
        ensemble_fiw=fiw_ens,
        single_fiw=single_fiw,
        weights=weights
    )
    meta_model.eval()

    out_dir = os.path.join(project_root, "weights", "active_ensemble")
    os.makedirs(out_dir, exist_ok=True)

    meta_ckpt_path = os.path.join(out_dir, "meta_ensemble_kinship.pt")
    meta_json_path = os.path.join(out_dir, "meta_ensemble_metadata.json")

    print(f"\nSaving Meta-Ensemble checkpoint to {meta_ckpt_path}...")
    torch.save(meta_model.state_dict(), meta_ckpt_path)

    metadata = {
        "model_name": "MetaEnsembleKinshipClassifier",
        "n_constituent_models": 11,
        "n_qubits": 8,
        "encoding_mode": "entangled",
        "projection_type": "quantum_inspired_attention",
        "domain_weights": {
            "ensemble_kinship_full": weights[0],
            "ensemble_kinship_fiw": weights[1],
            "best_checkpoint_fiw": weights[2]
        },
        "optimal_threshold": 0.5216,
        "source_checkpoints": [
            "weights/active_ensemble/ensemble_kinship_full.pt",
            "weights/active_ensemble/ensemble_kinship_fiw.pt",
            "outputs/fiw_retraining_improved/best_checkpoint.pt"
        ]
    }

    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Meta-Ensemble metadata to {meta_json_path}")
    print("\n[SUCCESS] Meta-Ensemble model created and bundled successfully!")

if __name__ == "__main__":
    main()





