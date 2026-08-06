# -*- coding: utf-8 -*-
"""
=================================================================================
  QUANTUM KINSHIP VERIFICATION -- STANDALONE META-ENSEMBLE DEPLOYMENT SCRIPT
=================================================================================

This script executes the complete end-to-end inference pipeline from raw face images
using ONLY the saved Meta-Ensemble model (meta_ensemble_kinship.pt).

Pipeline Execution:
  1. Loads raw face images (Image 1 & Image 2) from user input or CLI arguments.
  2. Extracts 512D facial embeddings dynamically using FaceNet.
  3. Prepares relationship one-hot context vector (FD, FS, MD, MS).
  4. Runs forward pass on MetaEnsembleKinshipClassifier (meta_ensemble_kinship.pt).
  5. Outputs probability, binary classification (KIN / NON-KIN), confidence & latency.

Usage:
  # CLI Mode:
  python scripts/inference/deploy_meta_predict.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg --relation fs

  # Interactive Prompt Mode:
  python scripts/inference/deploy_meta_predict.py
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# Project Root Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "training", "evaluation", "inference", "archive"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import (
    FaceFeatureExtractor,
    HybridKinshipClassifier,
    EnsembleKinshipClassifier,
    MetaEnsembleKinshipClassifier
)

RELATION_MAP = {
    "fd": (0, "Father-Daughter (FD)"),
    "1": (0, "Father-Daughter (FD)"),
    "father-daughter": (0, "Father-Daughter (FD)"),
    
    "fs": (1, "Father-Son (FS)"),
    "2": (1, "Father-Son (FS)"),
    "father-son": (1, "Father-Son (FS)"),

    "md": (2, "Mother-Daughter (MD)"),
    "3": (2, "Mother-Daughter (MD)"),
    "mother-daughter": (2, "Mother-Daughter (MD)"),

    "ms": (3, "Mother-Son (MS)"),
    "4": (3, "Mother-Son (MS)"),
    "mother-son": (3, "Mother-Son (MS)")
}

def load_meta_ensemble_only(project_root):
    """
    Loads ONLY meta_ensemble_kinship.pt and its metadata.
    """
    ckpt_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_kinship.pt")
    meta_json_path = os.path.join(project_root, "weights", "active_ensemble", "meta_ensemble_metadata.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] Meta-ensemble checkpoint not found at: {ckpt_path}\n"
                                "Please run 'python scripts/training/build_meta_ensemble.py' first.")

    optimal_threshold = 0.5216
    if os.path.exists(meta_json_path):
        with open(meta_json_path, "r") as f:
            metadata = json.load(f)
            optimal_threshold = metadata.get("optimal_threshold", 0.5216)

    # Initialize model structure for state dict loading
    m1 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m2 = EnsembleKinshipClassifier([HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention") for _ in range(5)])
    m3 = HybridKinshipClassifier(n_qubits=8, encoding_mode="entangled", projection_type="quantum_inspired_attention")
    
    meta_model = MetaEnsembleKinshipClassifier(m1, m2, m3, weights=(0.45, 0.35, 0.20))
    state_dict = torch.load(ckpt_path, map_location="cpu")
    meta_model.load_state_dict(state_dict)
    meta_model.eval()

    return meta_model, optimal_threshold

def predict_pair(meta_model, feature_extractor, img1_path, img2_path, rel_code, threshold=0.5216):
    """
    Runs complete end-to-end prediction from raw image files.
    """
    t0 = time.perf_counter()

    # Step 1: Extract 512D Embeddings from raw pixels
    emb1_np = feature_extractor.extract(img1_path)
    emb2_np = feature_extractor.extract(img2_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_model.to(device)

    emb1 = torch.tensor(emb1_np, dtype=torch.float32, device=device).unsqueeze(0)
    emb2 = torch.tensor(emb2_np, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Step 2: One-hot relation vector (1, 4)
    rels = F.one_hot(torch.tensor([rel_code], device=device), num_classes=4).float()

    # Step 3: Forward pass through Meta-Ensemble
    with torch.no_grad():
        prob = meta_model(emb1, emb2, rels).item()

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # Step 4: Decision & Confidence
    is_kin = prob >= threshold
    label = "KIN (Family Related)" if is_kin else "NON-KIN (Not Related)"
    
    # Confidence score calculation
    distance_from_thresh = abs(prob - threshold)
    confidence = min(99.9, 50.0 + (distance_from_thresh / 0.5) * 50.0)

    return {
        "probability": prob,
        "is_kin": is_kin,
        "label": label,
        "confidence": confidence,
        "threshold": threshold,
        "latency_ms": latency_ms
    }

def main():
    parser = argparse.ArgumentParser(description="Standalone Meta-Ensemble Kinship Verification Deployment Tool")
    parser.add_argument("--img1", type=str, help="Path to face image 1 (e.g. parent photo)")
    parser.add_argument("--img2", type=str, help="Path to face image 2 (e.g. child photo)")
    parser.add_argument("--relation", type=str, choices=["fd", "fs", "md", "ms", "1", "2", "3", "4"], help="Kinship relationship: fd (Father-Daughter), fs (Father-Son), md (Mother-Daughter), ms (Mother-Son)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (default: calibrated 0.5216)")
    args = parser.parse_args()

    print("=================================================================")
    print("  QUANTUM KINSHIP VERIFICATION — DEPLOYED META-MODEL SYSTEM")
    print("=================================================================")

    # Interactive input fallback if arguments are omitted
    img1_path = args.img1
    img2_path = args.img2
    relation_input = args.relation

    if not img1_path:
        img1_path = input("\nEnter path for Image 1 (Person 1 / Parent): ").strip().strip('"\'')
    if not img2_path:
        img2_path = input("Enter path for Image 2 (Person 2 / Child): ").strip().strip('"\'')

    if not relation_input:
        print("\nSelect Relationship Type:")
        print("  1) Father-Daughter (FD)")
        print("  2) Father-Son (FS)")
        print("  3) Mother-Daughter (MD)")
        print("  4) Mother-Son (MS)")
        relation_input = input("Selection (1-4 or fd/fs/md/ms): ").strip().lower()

    if relation_input not in RELATION_MAP:
        print(f"\n[ERROR] Invalid relation code '{relation_input}'. Defaulting to Father-Son (FS).")
        rel_code, rel_name = 1, "Father-Son (FS)"
    else:
        rel_code, rel_name = RELATION_MAP[relation_input]

    if not os.path.exists(img1_path):
        print(f"\n[ERROR] Image file does not exist: {img1_path}")
        sys.exit(1)
    if not os.path.exists(img2_path):
        print(f"\n[ERROR] Image file does not exist: {img2_path}")
        sys.exit(1)

    print("\n[1/3] Loading Deployed Meta-Ensemble Weights (meta_ensemble_kinship.pt)...")
    meta_model, default_thresh = load_meta_ensemble_only(project_root)
    threshold = args.threshold if args.threshold is not None else default_thresh

    print("[2/3] Initializing FaceNet Feature Extractor...")
    feature_extractor = FaceFeatureExtractor()

    print("[3/3] Executing End-to-End Prediction Pipeline...")
    res = predict_pair(meta_model, feature_extractor, img1_path, img2_path, rel_code, threshold=threshold)

    # Output Presentation
    print("\n" + "=" * 65)
    print("  VERIFICATION RESULT")
    print("=" * 65)
    print(f"  Image 1 Path        : {os.path.basename(img1_path)}")
    print(f"  Image 2 Path        : {os.path.basename(img2_path)}")
    print(f"  Relationship Type   : {rel_name}")
    print(f"  Decision Threshold  : {res['threshold']:.4f}")
    print("  -------------------------------------------------------------")
    print(f"  Kinship Probability : {res['probability'] * 100:.2f}%")
    print(f"  Classification      : >> {res['label']} <<")
    print(f"  System Confidence   : {res['confidence']:.1f}%")
    print(f"  Inference Latency   : {res['latency_ms']:.2f} ms")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
