# -*- coding: utf-8 -*-
"""
Bundle all fold model checkpoints into a single EnsembleKinshipClassifier .pt file.

Usage:
  python scripts/build_ensemble.py
"""

import os
import sys
import glob
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Robust project root resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "training", "evaluation", "inference", "archive"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier, EnsembleKinshipClassifier


def main():
    weights_dir = os.path.join(project_root, "weights")

    print("=" * 70)
    print("  BUILDING SINGLE ENSEMBLE MODEL FROM FOLD CHECKPOINTS")
    print("=" * 70)

    # Collect all valid improved model checkpoints
    # Use the main model + fold files (avoiding duplicates)
    fold_paths = sorted(glob.glob(os.path.join(weights_dir, "hybrid_kinship_improved_fold*_entangled.pt")))
    main_path = os.path.join(weights_dir, "hybrid_kinship_improved_entangled.pt")

    # The main .pt is a copy of the best fold, so use it as a replacement
    # for any corrupted fold file
    all_paths = fold_paths.copy()

    print(f"\n  Found {len(all_paths)} fold checkpoint(s) + 1 main checkpoint")

    # Load all valid models
    loaded_models = []
    loaded_names = []

    for path in all_paths:
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model = HybridKinshipClassifier(
                n_qubits=8,
                encoding_mode="entangled",
                projection_type="quantum_inspired_attention",
            )
            model.load_state_dict(state_dict)
            model.eval()
            loaded_models.append(model)
            loaded_names.append(os.path.basename(path))
            print(f"  [OK] Loaded: {os.path.basename(path)}")
        except Exception as e:
            print(f"  [SKIP] Corrupted: {os.path.basename(path)}")
            # Try to substitute with main checkpoint
            if os.path.exists(main_path) and main_path != path:
                try:
                    state_dict = torch.load(main_path, map_location="cpu", weights_only=True)
                    model = HybridKinshipClassifier(
                        n_qubits=8,
                        encoding_mode="entangled",
                        projection_type="quantum_inspired_attention",
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    loaded_models.append(model)
                    loaded_names.append(f"{os.path.basename(path)} (substituted from main)")
                    print(f"  [FIX] Substituted corrupted fold with: {os.path.basename(main_path)}")
                except Exception:
                    print(f"  [WARN] Could not substitute, skipping this fold.")

    if len(loaded_models) == 0:
        print("  [ERROR] No valid models found!")
        return

    print(f"\n  Total models in ensemble: {len(loaded_models)}")

    # Build ensemble
    ensemble = EnsembleKinshipClassifier(loaded_models)
    ensemble.eval()

    # Quick sanity check
    print("\n  Running sanity check...")
    dummy_emb1 = torch.randn(2, 512)
    dummy_emb2 = torch.randn(2, 512)
    dummy_rel = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)

    with torch.no_grad():
        out = ensemble(dummy_emb1, dummy_emb2, dummy_rel)
    print(f"  Sanity check passed! Output shape: {out.shape}, values: [{out.min():.4f}, {out.max():.4f}]")

    # Save as single state_dict file
    save_path = os.path.join(weights_dir, "ensemble_kinship_full.pt")
    torch.save(ensemble.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  Ensemble state_dict saved to: {save_path}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Contains: {len(loaded_models)} sub-models")

    # Save metadata
    import json
    meta = {
        "n_models": len(loaded_models),
        "n_qubits": 8,
        "encoding_mode": "entangled",
        "projection_type": "quantum_inspired_attention",
        "optimal_threshold": 0.5279678702354431,
        "source_models": loaded_names,
    }
    meta_path = os.path.join(weights_dir, "ensemble_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")

    print("\n" + "=" * 70)
    print("  ENSEMBLE MODEL BUILT SUCCESSFULLY!")
    print(f"  Use: ensemble_kinship_full.pt + ensemble_metadata.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
