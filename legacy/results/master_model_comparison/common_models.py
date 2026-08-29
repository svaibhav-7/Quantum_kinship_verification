# -*- coding: utf-8 -*-
"""
COMMON MODEL UTILITY for Master Model Comparison (4 Models x 12 Modules)

Loads the 4 candidate models into a unified registry.

Models:
  1. "ensemble_kinship_full"  -> EnsembleKinshipClassifier (5 folds, base)
  2. "meta_ensemble_kinship"  -> MetaEnsembleKinshipClassifier (11 models, hierarchical)
  3. "ensemble_kinship_fiw"    -> EnsembleKinshipClassifier (5 folds, FIW-tuned)
  4. "best_checkpoint"         -> HybridKinshipClassifier (single fine-tuned)

Usage:
    from common_models import get_models, MODEL_KEYS
    models = get_models()   # {key: model_instance}
"""

import os
import sys
import torch

_current = os.path.dirname(os.path.abspath(__file__))
research_root = os.path.dirname(_current)
project_root = os.path.dirname(research_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import (
    HybridKinshipClassifier,
    EnsembleKinshipClassifier,
    MetaEnsembleKinshipClassifier,
)

N_QUBITS = 8
ENCODING = "entangled"
PROJ = "quantum_inspired_attention"

MODEL_KEYS = [
    "ensemble_kinship_full",
    "meta_ensemble_kinship",
    "ensemble_kinship_fiw",
    "best_checkpoint",
]

MODEL_LABELS = {
    "ensemble_kinship_full": "Base Ensemble (5 folds)",
    "meta_ensemble_kinship": "Meta-Ensemble (11 models)",
    "ensemble_kinship_fiw": "FIW Ensemble (5 folds)",
    "best_checkpoint": "Best Checkpoint (1 model)",
}

MODEL_FILES = {
    "ensemble_kinship_full": os.path.join(project_root, "weights", "active_ensemble", "ensemble_kinship_full.pt"),
    "meta_ensemble_kinship": os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt"),
    "ensemble_kinship_fiw": os.path.join(project_root, "weights", "secondary_active_ensemble", "ensemble_kinship_fiw.pt"),
    "best_checkpoint": os.path.join(project_root, "outputs", "fiw_retraining_improved", "best_checkpoint.pt"),
}


def _single():
    return HybridKinshipClassifier(n_qubits=N_QUBITS, encoding_mode=ENCODING, projection_type=PROJ)


def _ensemble(n=5):
    return EnsembleKinshipClassifier([_single() for _ in range(n)])


def _build_model(key):
    if key == "ensemble_kinship_full":
        return _ensemble(5)
    if key == "meta_ensemble_kinship":
        m1 = _ensemble(5)
        m2 = _ensemble(5)
        m3 = _single()
        return MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
    if key == "ensemble_kinship_fiw":
        return _ensemble(5)
    if key == "best_checkpoint":
        return _single()
    raise ValueError(f"Unknown model key: {key}")


def load_model(key, device=None):
    """Build the correct architecture and load weights for a model key."""
    model = _build_model(key)
    path = MODEL_FILES[key]
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_models(device=None):
    """Return {model_key: loaded model} for all 4 models."""
    models = {}
    for key in MODEL_KEYS:
        print(f"  [LOAD] {key} ...")
        models[key] = load_model(key, device=device)
    return models


if __name__ == "__main__":
    ms = get_models()
    for k, m in ms.items():
        n = sum(p.numel() for p in m.parameters())
        print(f"{k}: {MODEL_LABELS[k]}, params={n}")
