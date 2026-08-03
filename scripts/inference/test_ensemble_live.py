# -*- coding: utf-8 -*-
"""
=============================================================================
  QUANTUM KINSHIP VERIFICATION -- COMPREHENSIVE ENSEMBLE EVALUATION & 
  LIVE USER PREDICTION SCRIPT
=============================================================================

Evaluates the trained 5-fold ensemble model on ALL available KinFaceW-I pairs
with full from-scratch FaceNet embedding extraction (no cached embeddings).
Also supports live user-provided image pair predictions.

Usage:
  # Full dataset evaluation (all KinFaceW-I pairs, from scratch):
  python scripts/test_ensemble_live.py --mode evaluate

  # Predict on two user-provided images:
  python scripts/test_ensemble_live.py --mode predict --img1 path/to/face1.jpg --img2 path/to/face2.jpg

  # Predict with a specific relation type:
  python scripts/test_ensemble_live.py --mode predict --img1 face1.jpg --img2 face2.jpg --relation fd

  # Both: run full evaluation then predict user images:
  python scripts/test_ensemble_live.py --mode both --img1 face1.jpg --img2 face2.jpg
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

# Force UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Robust project root resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) in ["scripts", "training", "evaluation", "inference", "archive"]:
    current_dir = os.path.dirname(current_dir)
project_root = current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from src.data_loaders import get_relation_category, load_kinfacew_pairs
from src.models_improved import (
    EnsembleKinshipClassifier,
    FaceFeatureExtractor,
    HybridKinshipClassifier,
)


# =============================================================================
# Constants
# =============================================================================

OPTIMAL_THRESHOLD = 0.5279678702354431
RELATION_LABELS = {0: "Father-Daughter (FD)", 1: "Father-Son (FS)",
                   2: "Mother-Daughter (MD)", 3: "Mother-Son (MS)"}
RELATION_SHORT = {0: "FD", 1: "FS", 2: "MD", 3: "MS"}


# =============================================================================
# Model Loading
# =============================================================================

def find_weight_file(filename, project_root):
    """Search for weight/cache file across weights subdirectories."""
    search_dirs = [
        os.path.join(project_root, "weights", "active_ensemble"),
        os.path.join(project_root, "weights", "folds"),
        os.path.join(project_root, "weights", "caches"),
        os.path.join(project_root, "weights"),
        os.path.join(project_root, "weights", "archive"),
    ]
    for s_dir in search_dirs:
        candidate = os.path.join(s_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(project_root, "weights", filename)


def load_ensemble(weights_dir):
    """
    Load the ensemble model. Tries the single bundled file first,
    falls back to loading individual fold checkpoints.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(current_dir) in ["scripts", "training", "evaluation", "inference", "archive"]:
        current_dir = os.path.dirname(current_dir)
    project_root = current_dir

    # Try single bundled ensemble file first
    ensemble_path = find_weight_file("ensemble_kinship_full.pt", project_root)
    meta_path = find_weight_file("ensemble_metadata.json", project_root)
    if os.path.exists(ensemble_path) and os.path.exists(meta_path):
        print(f"  Loading single bundled ensemble from: {os.path.basename(ensemble_path)}")
        try:
            import json as _json
            with open(meta_path) as f:
                meta = _json.load(f)
            n_models = meta["n_models"]
            # Build the ensemble structure, then load weights
            sub_models = [
                HybridKinshipClassifier(
                    n_qubits=meta.get("n_qubits", 8),
                    encoding_mode=meta.get("encoding_mode", "entangled"),
                    projection_type=meta.get("projection_type", "quantum_inspired_attention"),
                )
                for _ in range(n_models)
            ]
            ensemble_model = EnsembleKinshipClassifier(sub_models)
            state_dict = torch.load(ensemble_path, map_location="cpu", weights_only=True)
            ensemble_model.load_state_dict(state_dict)
            ensemble_model.eval()
            print(f"  Loaded ensemble with {n_models} sub-models in a single file!")
            return [(ensemble_model, f"ensemble_kinship_full.pt ({n_models} models)")]
        except Exception as e:
            print(f"  [WARN] Failed to load bundled ensemble: {e}")
            print(f"  Falling back to individual fold files...")

    # Fallback: load individual checkpoints from weights/folds/ or weights/
    folds_dir = os.path.join(project_root, "weights", "folds")
    model_paths = sorted(glob.glob(os.path.join(folds_dir, "hybrid_kinship_improved*.pt"))) or sorted(glob.glob(os.path.join(weights_dir, "hybrid_kinship_improved*.pt")))

    if len(model_paths) == 0:
        print(f"  [ERROR] No model weights found in {weights_dir}")
        sys.exit(1)

    models = []
    for path in model_paths:
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model = HybridKinshipClassifier(
                n_qubits=8,
                encoding_mode="entangled",
                projection_type="quantum_inspired_attention",
            )
            model.load_state_dict(state_dict)
            model.eval()
            models.append((model, os.path.basename(path)))
        except Exception as e:
            print(f"  [WARN] Skipping corrupted checkpoint: {os.path.basename(path)} ({e})")

    return models


def ensemble_predict(models, emb1_t, emb2_t, rel_t):
    """Run forward pass through all models and return averaged fidelity score."""
    scores = []
    with torch.no_grad():
        for model, _ in models:
            score = model(emb1_t, emb2_t, rel_t).item()
            scores.append(score)
    return float(np.mean(scores)), scores


# =============================================================================
# Full Dataset Evaluation (From Scratch)
# =============================================================================

def run_full_evaluation(models, extractor, kfw1_dir):
    """
    Evaluate the ensemble on ALL KinFaceW-I pairs with from-scratch
    FaceNet embedding extraction (no cache). Simulates real deployment.
    """
    print("\n" + "=" * 72)
    print("  PHASE 1: FULL DATASET EVALUATION (FROM SCRATCH -- NO CACHE)")
    print("=" * 72)

    # Load all pairs
    pairs = load_kinfacew_pairs(kfw1_dir)
    if not pairs:
        print("  [ERROR] Could not load KinFaceW-I pairs.")
        return

    total_pairs = len(pairs)
    kin_count = sum(1 for p in pairs if p[2] == 1)
    nonkin_count = total_pairs - kin_count
    print(f"\n  Dataset: KinFaceW-I")
    print(f"  Total Pairs:     {total_pairs}")
    print(f"  Kin Pairs:       {kin_count}")
    print(f"  Non-Kin Pairs:   {nonkin_count}")
    print(f"  Ensemble Models: {len(models)}")
    print(f"  Threshold:       {OPTIMAL_THRESHOLD:.4f}")
    print("-" * 72)

    # Extract embeddings from scratch for every pair
    print("\n  [Step 1/3] Extracting FaceNet embeddings from scratch...")
    t_start = time.perf_counter()

    all_labels = []
    all_preds = []
    all_relations = []
    per_relation_results = {0: {"correct": 0, "total": 0},
                            1: {"correct": 0, "total": 0},
                            2: {"correct": 0, "total": 0},
                            3: {"correct": 0, "total": 0}}

    batch_report_interval = max(1, total_pairs // 10)

    for idx, (p1_path, p2_path, label, rel_str) in enumerate(pairs):
        # Progress reporting
        if (idx + 1) % batch_report_interval == 0 or idx == 0:
            pct = ((idx + 1) / total_pairs) * 100
            print(f"    Processing pair {idx+1}/{total_pairs} ({pct:.0f}%)...")

        # Extract embeddings from scratch (no cache)
        try:
            emb1 = extractor.extract(p1_path)
            emb2 = extractor.extract(p2_path)
        except Exception as e:
            print(f"    [WARN] Skipping pair {idx+1} ({os.path.basename(p1_path)}, "
                  f"{os.path.basename(p2_path)}): {e}")
            continue

        emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
        emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)

        cat = get_relation_category(rel_str, p1_path)
        rel_vec = [0.0] * 4
        rel_vec[cat] = 1.0
        rel_t = torch.tensor([rel_vec], dtype=torch.float32)

        # Ensemble prediction
        avg_score, _ = ensemble_predict(models, emb1_t, emb2_t, rel_t)

        all_labels.append(label)
        all_preds.append(avg_score)
        all_relations.append(cat)

        # Per-relation tracking
        is_correct = (avg_score >= OPTIMAL_THRESHOLD) == (label == 1)
        per_relation_results[cat]["total"] += 1
        if is_correct:
            per_relation_results[cat]["correct"] += 1

    t_extract = time.perf_counter() - t_start
    print(f"\n  [Step 1/3] Embedding extraction + inference completed in {t_extract:.1f}s")

    # Compute metrics
    print("\n  [Step 2/3] Computing classification metrics...")
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)

    # Standard threshold (0.5)
    pred_labels_std = (preds_np >= 0.5).astype(float)
    acc_std = accuracy_score(labels_np, pred_labels_std) * 100

    # Optimal threshold
    pred_labels_opt = (preds_np >= OPTIMAL_THRESHOLD).astype(float)
    acc_opt = accuracy_score(labels_np, pred_labels_opt) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, pred_labels_opt, average="binary", zero_division=0
    )

    try:
        fpr, tpr, _ = roc_curve(labels_np, preds_np)
        roc_auc_val = auc(fpr, tpr)
    except ValueError:
        roc_auc_val = 0.5

    tn, fp, fn, tp = confusion_matrix(labels_np, pred_labels_opt).ravel()

    # Youden's J from scratch
    try:
        fpr_j, tpr_j, thresholds_j = roc_curve(labels_np, preds_np)
        youden_j = tpr_j - fpr_j
        best_idx = np.argmax(youden_j)
        computed_threshold = thresholds_j[best_idx]
        computed_youden = youden_j[best_idx]
    except Exception:
        computed_threshold = 0.5
        computed_youden = 0.0

    # Print results
    print("\n  [Step 3/3] RESULTS")
    print("\n" + "=" * 72)
    print("  FULL EVALUATION RESULTS -- QUANTUM KINSHIP ENSEMBLE MODEL")
    print("=" * 72)

    print(f"\n  [Overall Performance]")
    print(f"  * Pairs Evaluated:           {len(labels_np)}")
    print(f"  * Accuracy (t=0.500):        {acc_std:.2f}%")
    print(f"  * Accuracy (t={OPTIMAL_THRESHOLD:.3f}):     {acc_opt:.2f}%")
    print(f"  * Computed Optimal t:        {computed_threshold:.4f} (Youden J = {computed_youden:.4f})")
    print(f"  * ROC-AUC:                   {roc_auc_val:.4f}")
    print(f"  * Precision:                 {precision*100:.2f}%")
    print(f"  * Recall (Sensitivity):      {recall*100:.2f}%")
    print(f"  * F1-Score:                  {f1*100:.2f}%")
    print(f"  * Confusion Matrix:          TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  * Inference Time:            {t_extract:.1f}s ({t_extract/len(labels_np)*1000:.1f}ms/pair)")

    print(f"\n  [Per-Relation Breakdown]")
    print(f"  {'Relation':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*55}")
    for cat_id in sorted(per_relation_results.keys()):
        stats = per_relation_results[cat_id]
        if stats["total"] > 0:
            rel_acc = (stats["correct"] / stats["total"]) * 100
            print(f"  {RELATION_LABELS[cat_id]:<25} {stats['correct']:>8} {stats['total']:>8} {rel_acc:>9.2f}%")

    # Kin vs Non-Kin score separation analysis
    kin_scores = preds_np[labels_np == 1]
    nonkin_scores = preds_np[labels_np == 0]
    print(f"\n  [Score Distribution Analysis]")
    print(f"  * Kin Pairs Mean Fidelity:     {np.mean(kin_scores)*100:.2f}% (std: {np.std(kin_scores)*100:.2f}%)")
    print(f"  * Non-Kin Pairs Mean Fidelity: {np.mean(nonkin_scores)*100:.2f}% (std: {np.std(nonkin_scores)*100:.2f}%)")
    print(f"  * Separation Gap:             {(np.mean(kin_scores) - np.mean(nonkin_scores))*100:.2f} percentage points")
    print("=" * 72)

    return {
        "accuracy_standard": acc_std,
        "accuracy_optimal": acc_opt,
        "roc_auc": roc_auc_val,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "computed_threshold": float(computed_threshold),
        "per_relation": {
            RELATION_SHORT[k]: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": (v["correct"] / v["total"] * 100) if v["total"] > 0 else 0
            }
            for k, v in per_relation_results.items()
        },
        "kin_mean_fidelity": float(np.mean(kin_scores)),
        "nonkin_mean_fidelity": float(np.mean(nonkin_scores)),
        "inference_time_seconds": t_extract,
    }


# =============================================================================
# User Image Prediction (From Scratch)
# =============================================================================

def predict_user_images(models, extractor, img1_path, img2_path):
    """
    Predict kinship between two user-provided face images.
    Extracts FaceNet embeddings from scratch (no cache).
    Tests all 4 relation types, averages scores, and returns a single
    binary KIN / NON-KIN decision.
    """
    print("\n" + "=" * 72)
    print("  LIVE PREDICTION -- BINARY KINSHIP CLASSIFIER")
    print("=" * 72)

    # Validate images exist
    if not os.path.exists(img1_path):
        print(f"  [ERROR] Image not found: {img1_path}")
        return
    if not os.path.exists(img2_path):
        print(f"  [ERROR] Image not found: {img2_path}")
        return

    print(f"\n  Face 1: {os.path.abspath(img1_path)}")
    print(f"  Face 2: {os.path.abspath(img2_path)}")
    print("-" * 72)

    # Extract embeddings from scratch
    print("\n  [1/2] Extracting FaceNet embeddings from scratch...")
    t0 = time.perf_counter()
    try:
        emb1 = extractor.extract(img1_path)
        emb2 = extractor.extract(img2_path)
    except Exception as e:
        print(f"  [ERROR] Embedding extraction failed: {e}")
        return
    t_emb = time.perf_counter() - t0
    print(f"  Embeddings extracted in {t_emb:.2f}s")

    emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
    emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)

    # Prediction: test all 4 relation types, average the scores
    print("\n  [2/2] Running ensemble quantum SWAP-test inference...")
    print("         (Averaging across all 4 relation conditionings)\n")

    relation_scores = []
    for cat in range(4):
        rel_vec = [0.0] * 4
        rel_vec[cat] = 1.0
        rel_t = torch.tensor([rel_vec], dtype=torch.float32)

        avg_score, _ = ensemble_predict(models, emb1_t, emb2_t, rel_t)
        relation_scores.append(avg_score)
        print(f"    {RELATION_LABELS[cat]:<25}: {avg_score*100:.2f}%")

    # Final binary decision: average across all relation types
    overall_fidelity = float(np.mean(relation_scores))
    decision = "RELATED (KIN)" if overall_fidelity >= OPTIMAL_THRESHOLD else "NOT RELATED (NON-KIN)"
    confidence = abs(overall_fidelity - OPTIMAL_THRESHOLD) / OPTIMAL_THRESHOLD * 100

    print(f"\n  ================================================")
    print(f"  |                                              |")
    print(f"  |  Overall Quantum Fidelity:  {overall_fidelity*100:>6.2f}%          |")
    print(f"  |  Decision Threshold:        {OPTIMAL_THRESHOLD*100:>6.2f}%          |")
    print(f"  |  Confidence:                {confidence:>6.1f}%          |")
    print(f"  |                                              |")
    print(f"  |  PREDICTION:  {decision:<30} |")
    print(f"  |                                              |")
    print(f"  ================================================")
    print("=" * 72)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Kinship Verification -- Ensemble Binary Classifier & Live Prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["evaluate", "predict", "both"],
        default="evaluate",
        help="'evaluate' = full dataset test, 'predict' = user image prediction, 'both' = both",
    )
    parser.add_argument("--img1", type=str, default=None, help="Path to first face image (for predict mode)")
    parser.add_argument("--img2", type=str, default=None, help="Path to second face image (for predict mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "weights")
    kfw1_dir = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")
    results_dir = os.path.join(project_root, "results", "training_metrics")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("  QUANTUM KINSHIP VERIFICATION -- COMPREHENSIVE ENSEMBLE TESTING")
    print("  (All embeddings extracted from scratch -- deployment simulation)")
    print("=" * 72)

    # 1. Initialize FaceNet Feature Extractor (from scratch)
    print("\n[INIT] Initializing FaceNet Feature Extractor...")
    extractor = FaceFeatureExtractor()

    # 2. Load Ensemble Models
    print("\n[INIT] Loading Ensemble Model Checkpoints...")
    models = load_ensemble(weights_dir)
    print(f"  Loaded {len(models)} model(s):")
    for _, name in models:
        print(f"    - {name}")

    # 3. Run evaluation mode
    if args.mode in ("evaluate", "both"):
        eval_results = run_full_evaluation(models, extractor, kfw1_dir)

        if eval_results:
            # Save results JSON
            eval_json_path = os.path.join(results_dir, "ensemble_live_evaluation.json")
            with open(eval_json_path, "w") as f:
                json.dump(eval_results, f, indent=2)
            print(f"\n  Evaluation results saved to: {eval_json_path}")

    # 4. Run predict mode
    if args.mode in ("predict", "both"):
        if not args.img1 or not args.img2:
            print("\n  [ERROR] --img1 and --img2 are required for predict mode.")
            print("  Usage: python scripts/test_ensemble_live.py --mode predict --img1 face1.jpg --img2 face2.jpg")
            sys.exit(1)
        predict_user_images(models, extractor, args.img1, args.img2)

    print("\n  DONE!")


if __name__ == "__main__":
    main()
