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

# Project root resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "training", "evaluation", "inference", "tests"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier, MetaEnsembleKinshipClassifier

def load_base_ensemble(project_root):
    path = os.path.join(project_root, "weights", "active_ensemble", "ensemble_kinship_full.pt")
    models = [HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)]
    ens = EnsembleKinshipClassifier(models)
    state = torch.load(path, map_location="cpu")
    ens.load_state_dict(state)
    return ens

def load_fiw_ensemble(project_root):
    path = os.path.join(project_root, "weights", "secondary_active_ensemble", "ensemble_kinship_fiw.pt")
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
    
    # Construct Meta-Ensemble with Domain-Weighted Soft Voting (0.45, 0.35, 0.20)
    meta_model = MetaEnsembleKinshipClassifier(
        ensemble_full=base_ens,
        ensemble_fiw=fiw_ens,
        single_fiw=single_fiw,
        weights=(0.45, 0.35, 0.20)
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
            "ensemble_kinship_full": 0.45,
            "ensemble_kinship_fiw": 0.35,
            "best_checkpoint_fiw": 0.20
        },
        "optimal_threshold": 0.5216,
        "source_checkpoints": [
            "weights/active_ensemble/ensemble_kinship_full.pt",
            "weights/secondary_active_ensemble/ensemble_kinship_fiw.pt",
            "outputs/fiw_retraining_improved/best_checkpoint.pt"
        ]
    }
    
    with open(meta_json_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved Meta-Ensemble metadata to {meta_json_path}")
    print("\n[SUCCESS] Meta-Ensemble model created and bundled successfully!")

if __name__ == "__main__":
    main()
