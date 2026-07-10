# -*- coding: utf-8 -*-
"""
Predict kinship between two user-provided face images.

Key design decisions for accurate inference:
  - Ensemble averaging across all fold models (not just a single fold)
  - Optimal threshold from Youden's J (auto-loaded from training results)
  - Same preprocessing pipeline as training (no MTCNN -- avoids domain gap)
  - Try-all-relations mode: test all 4 relation types and report best match

Usage:
  python scripts/predict_user_images.py path/to/img1.jpg path/to/img2.jpg
  python scripts/predict_user_images.py img1.jpg img2.jpg --relation fd
  python scripts/predict_user_images.py img1.jpg img2.jpg --no-plot
"""

import argparse
import glob
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loaders import get_relation_category
from src.models import FaceFeatureExtractor, HybridKinshipClassifier

# --- Constants ----------------------------------------------------------------

RELATION_MAP = {
    "fd": ("Father-Daughter", 0),
    "fs": ("Father-Son", 1),
    "md": ("Mother-Daughter", 2),
    "ms": ("Mother-Son", 3),
    "father-daughter": ("Father-Daughter", 0),
    "father-son": ("Father-Son", 1),
    "mother-daughter": ("Mother-Daughter", 2),
    "mother-son": ("Mother-Son", 3),
}

ALL_RELATIONS = ["fd", "fs", "md", "ms"]
ALL_RELATION_NAMES = ["Father-Daughter", "Father-Son", "Mother-Daughter", "Mother-Son"]


# --- Model Loading ------------------------------------------------------------


def detect_checkpoint_config(state_dict):
    """Infer model config from a saved HybridKinshipClassifier state dict."""
    n_qubits = 8
    encoding_mode = "product"
    projection_type = "cross_attention"

    if "ent_params" in state_dict:
        n_qubits = int(state_dict["ent_params"].shape[0])
        if torch.any(torch.abs(state_dict["ent_params"]) > 1e-8):
            encoding_mode = "entangled"
    elif "ent_params1" in state_dict and "ent_params2" in state_dict:
        n_qubits = int(state_dict["ent_params1"].shape[0])
        encoding_mode = "entangled"

    if any("cross_attn" in k for k in state_dict):
        projection_type = "cross_attention"
    elif any("projection_net.projection" in k for k in state_dict):
        projection_type = "simple"

    for key, value in state_dict.items():
        if key.endswith(".bias") and "projection" in key and hasattr(value, "shape"):
            candidate = int(value.shape[0])
            if 1 <= candidate <= 32:
                n_qubits = candidate

    return n_qubits, encoding_mode, projection_type


def load_ensemble_models(weights_dir):
    """
    Load all available fold models for ensemble inference.

    Falls back to a single model if fold weights are not found.
    Returns a list of (model, config_dict) tuples.
    """
    models = []

    # Try fold models first (ensemble)
    fold_pattern = os.path.join(weights_dir, "hybrid_kinship_fold*.pt")
    fold_paths = sorted(glob.glob(fold_pattern))

    if len(fold_paths) >= 2:
        print(f"  Found {len(fold_paths)} fold models -- using ensemble inference")
        for path in fold_paths:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            n_qubits, enc_mode, proj_type = detect_checkpoint_config(state_dict)
            model = HybridKinshipClassifier(
                n_qubits=n_qubits,
                encoding_mode=enc_mode,
                projection_type=proj_type,
            )
            model.load_state_dict(state_dict)
            model.eval()
            models.append(
                (
                    model,
                    {
                        "path": os.path.basename(path),
                        "n_qubits": n_qubits,
                        "encoding_mode": enc_mode,
                        "projection_type": proj_type,
                    },
                )
            )
    else:
        # Fallback: single best model
        for name in [
            "hybrid_kinship_entangled.pt",
            "hybrid_kinship.pt",
        ]:
            single_path = os.path.join(weights_dir, name)
            if os.path.exists(single_path):
                print(f"  No fold models found -- using single model: {name}")
                state_dict = torch.load(
                    single_path, map_location="cpu", weights_only=True
                )
                n_qubits, enc_mode, proj_type = detect_checkpoint_config(state_dict)
                model = HybridKinshipClassifier(
                    n_qubits=n_qubits,
                    encoding_mode=enc_mode,
                    projection_type=proj_type,
                )
                model.load_state_dict(state_dict)
                model.eval()
                models.append(
                    (
                        model,
                        {
                            "path": name,
                            "n_qubits": n_qubits,
                            "encoding_mode": enc_mode,
                            "projection_type": proj_type,
                        },
                    )
                )
                break

    if len(models) == 0:
        raise FileNotFoundError(
            f"No model weights found in {weights_dir}. "
            "Please train first with: python scripts/train_hybrid.py"
        )

    return models


def load_optimal_threshold(project_root):
    """
    Load the optimal threshold from training results.

    Priority order:
      1. Ensemble optimal threshold from fold_results.json
      2. Mean of per-fold optimal thresholds
      3. Default 0.5
    """
    results_paths = [
        os.path.join(
            project_root, "results", "training_metrics", "fold_results.json"
        ),
        os.path.join(
            project_root,
            "results",
            "training_metrics",
            "fold_results_entangled.json",
        ),
    ]

    for path in results_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Prefer ensemble threshold
            if "ensemble" in data and "optimal_threshold" in data["ensemble"]:
                thresh = float(data["ensemble"]["optimal_threshold"])
                print(f"  Loaded ensemble optimal threshold: {thresh:.4f}")
                return thresh

            # Fallback to mean of per-fold thresholds
            if "folds" in data:
                thresholds = [
                    float(fold["optimal_threshold"])
                    for fold in data["folds"]
                    if "optimal_threshold" in fold
                ]
                if thresholds:
                    thresh = float(np.mean(thresholds))
                    print(f"  Loaded mean fold threshold: {thresh:.4f}")
                    return thresh

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Could not read threshold from {path}: {e}")

    print("  [INFO] No saved threshold found -- using default 0.5")
    return 0.5


# --- Inference ----------------------------------------------------------------


def make_relation_tensor(relation_idx):
    """Create a one-hot (1, 4) relation tensor from a category index."""
    one_hot = [0.0] * 4
    one_hot[relation_idx] = 1.0
    return torch.tensor([one_hot], dtype=torch.float32)


def ensemble_predict(models, emb1_t, emb2_t, rel_t):
    """
    Run ensemble inference: average predictions across all fold models.

    Returns:
        mean_score: float, averaged fidelity across models
        per_model_scores: list of floats, individual model scores
    """
    scores = []
    for model, _ in models:
        with torch.no_grad():
            score = model(emb1_t, emb2_t, rel_t).item()
            scores.append(score)
    return float(np.mean(scores)), scores


def predict_all_relations(models, emb1_t, emb2_t):
    """
    Try all 4 relation types and return scores for each.

    Returns a list of (relation_name, relation_code, mean_score, per_model_scores).
    """
    results = []
    for i, (rel_code, rel_name) in enumerate(
        zip(ALL_RELATIONS, ALL_RELATION_NAMES)
    ):
        rel_t = make_relation_tensor(i)
        mean_score, per_model = ensemble_predict(models, emb1_t, emb2_t, rel_t)
        results.append((rel_name, rel_code, mean_score, per_model))
    return results


def confidence_label(score, threshold):
    """Map a fidelity score to a human-readable confidence label."""
    if score >= threshold + 0.15:
        return "HIGH"
    elif score >= threshold:
        return "MODERATE"
    elif score >= threshold - 0.10:
        return "LOW (borderline)"
    else:
        return "UNLIKELY"


# --- Visualization ------------------------------------------------------------


def setup_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0F172A",
            "axes.facecolor": "#1E293B",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#E2E8F0",
            "text.color": "#E2E8F0",
            "xtick.color": "#E2E8F0",
            "ytick.color": "#E2E8F0",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "figure.dpi": 150,
        }
    )


def create_prediction_plot(
    img1_path,
    img2_path,
    relation_results,
    best_idx,
    threshold,
    n_models,
    save_path=None,
    show=True,
):
    """Create a rich visualization of the prediction results."""
    setup_plot_style()

    best_name, best_code, best_score, _ = relation_results[best_idx]
    is_kin = best_score >= threshold

    fig = plt.figure(figsize=(12, 6))

    # Left panel: Face images
    ax1 = fig.add_axes([0.02, 0.15, 0.28, 0.7])
    ax2 = fig.add_axes([0.32, 0.15, 0.28, 0.7])

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax1.set_title("Image 1", fontsize=11, fontweight="bold", pad=8)
    ax2.set_title("Image 2", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")
    ax2.axis("off")

    # Right panel: Scores bar chart
    ax3 = fig.add_axes([0.68, 0.15, 0.28, 0.7])
    rel_names_short = ["FD", "FS", "MD", "MS"]
    scores = [r[2] for r in relation_results]
    colors = []
    for i, s in enumerate(scores):
        if i == best_idx:
            colors.append("#38BDF8" if is_kin else "#EF4444")
        else:
            colors.append("#475569")

    bars = ax3.barh(rel_names_short, scores, color=colors, edgecolor="none", height=0.6)
    ax3.axvline(x=threshold, color="#F59E0B", linestyle="--", linewidth=1.5, alpha=0.8)
    ax3.set_xlim(0, 1.05)
    ax3.set_xlabel("Fidelity Score", fontsize=10)
    ax3.set_title("Per-Relation Scores", fontsize=11, fontweight="bold", pad=8)

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax3.text(
            width + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=9,
            color="#E2E8F0",
        )

    # Title
    color = "#34D399" if is_kin else "#EF4444"
    verdict = "KIN" if is_kin else "NON-KIN"
    conf = confidence_label(best_score, threshold)
    fig.suptitle(
        f"{verdict}  |  Best: {best_name} ({best_score:.3f})  |  "
        f"Confidence: {conf}  |  Ensemble: {n_models} models  |  t={threshold:.3f}",
        color=color,
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    fig.patch.set_edgecolor(color)
    fig.patch.set_linewidth(3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# --- CLI ----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict kinship between two face images using the trained ensemble."
    )
    parser.add_argument("image1", help="Path to the first face image")
    parser.add_argument("image2", help="Path to the second face image")
    parser.add_argument(
        "--relation",
        choices=list(RELATION_MAP.keys()) + ["auto"],
        default="auto",
        help=(
            "Relation type for prediction. "
            "'auto' (default) tests all 4 relations and reports the best match."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        default=None,
        help="Directory containing model weights (default: weights/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (default: auto-load from training results)",
    )
    parser.add_argument(
        "--fallback-resnet",
        action="store_true",
        help="Force ResNet-18 fallback instead of FaceNet.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not open a matplotlib window."
    )
    parser.add_argument(
        "--save-plot",
        default=None,
        help="Optional output path for the prediction plot.",
    )
    parser.add_argument(
        "--qiskit",
        action="store_true",
        help="Also run Qiskit AerSimulator verification (slow).",
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="Qiskit AerSimulator shots."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-model scores for diagnostics."
    )
    return parser.parse_args()


# --- Main ---------------------------------------------------------------------


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = (
        args.weights_dir
        if args.weights_dir and os.path.isabs(args.weights_dir)
        else os.path.join(project_root, args.weights_dir or "weights")
    )

    # Validate images
    for img_arg, label in [(args.image1, "Image 1"), (args.image2, "Image 2")]:
        if not os.path.exists(img_arg):
            # Try relative to project root
            alt = os.path.join(project_root, img_arg)
            if not os.path.exists(alt):
                raise FileNotFoundError(f"{label} not found: {img_arg}")

    img1_path = (
        args.image1
        if os.path.exists(args.image1)
        else os.path.join(project_root, args.image1)
    )
    img2_path = (
        args.image2
        if os.path.exists(args.image2)
        else os.path.join(project_root, args.image2)
    )

    print("=" * 68)
    print("   QUANTUM KINSHIP VERIFICATION - ENSEMBLE PREDICTOR")
    print("=" * 68)

    # -- Step 1: Load models ----
    print("\n[1/4] Loading ensemble models...")
    models = load_ensemble_models(weights_dir)
    n_models = len(models)
    config = models[0][1]
    print(
        f"  [OK] {n_models} model(s) loaded | "
        f"{config['encoding_mode']} encoding | "
        f"{config['projection_type']} projection | "
        f"{config['n_qubits']} qubits"
    )

    # -- Step 2: Load threshold ----
    print("\n[2/4] Loading decision threshold...")
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  Using user-specified threshold: {threshold:.4f}")
    else:
        threshold = load_optimal_threshold(project_root)

    # -- Step 3: Extract embeddings --
    # NOTE: Use the same pipeline as training -- no MTCNN, just direct
    # FaceNet extraction. The training data images were already cropped
    # faces, so user images should ideally also be face crops.
    print("\n[3/4] Extracting facial embeddings...")
    t0 = time.perf_counter()
    extractor = FaceFeatureExtractor(use_resnet_fallback=args.fallback_resnet)

    emb1 = extractor.extract(img1_path)
    emb2 = extractor.extract(img2_path)
    feature_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] Feature extraction: {feature_ms:.1f} ms")

    emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
    emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)

    # -- Step 4: Run inference ----
    print("\n[4/4] Running ensemble inference...")
    t0 = time.perf_counter()

    if args.relation == "auto":
        # Test all 4 relation types and find the best match
        relation_results = predict_all_relations(models, emb1_t, emb2_t)
        best_idx = int(np.argmax([r[2] for r in relation_results]))
        best_name, best_code, best_score, best_per_model = relation_results[best_idx]
    else:
        # Use specified relation
        rel_name, rel_idx = RELATION_MAP[args.relation]
        rel_t = make_relation_tensor(rel_idx)
        mean_score, per_model = ensemble_predict(models, emb1_t, emb2_t, rel_t)
        relation_results = [(rel_name, args.relation, mean_score, per_model)]
        best_idx = 0
        best_name, best_code, best_score, best_per_model = relation_results[0]

    inference_ms = (time.perf_counter() - t0) * 1000

    # -- Qiskit verification (optional) ----
    qiskit_score = None
    qiskit_ms = None
    if args.qiskit:
        print("  Running Qiskit AerSimulator verification...")
        t0 = time.perf_counter()
        rel_t = make_relation_tensor(RELATION_MAP.get(best_code, ("", 0))[1])
        qiskit_scores = []
        for model, _ in models:
            with torch.no_grad():
                qs = model.forward_qiskit(
                    emb1_t, emb2_t, rel_t, shots=args.shots
                ).item()
                qiskit_scores.append(qs)
        qiskit_score = float(np.mean(qiskit_scores))
        qiskit_ms = (time.perf_counter() - t0) * 1000

    # -- Results -----
    is_kin = best_score >= threshold
    verdict = "KIN" if is_kin else "NON-KIN"
    conf = confidence_label(best_score, threshold)

    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)
    print(f"  Prediction:            {verdict}")
    print(f"  Confidence:            {conf}")
    print(f"  Best relation match:   {best_name} ({best_code})")
    print(f"  Fidelity score:        {best_score:.4f} ({best_score * 100:.2f}%)")
    print(f"  Decision threshold:    {threshold:.4f}")
    print(f"  Ensemble size:         {n_models} model(s)")
    print(f"  Inference time:        {inference_ms:.1f} ms")

    if qiskit_score is not None:
        qiskit_verdict = "KIN" if qiskit_score >= threshold else "NON-KIN"
        print(f"  Qiskit fidelity:       {qiskit_score:.4f} ({qiskit_score * 100:.2f}%)")
        print(f"  Qiskit verdict:        {qiskit_verdict}")
        print(f"  Qiskit time:           {qiskit_ms:.1f} ms ({args.shots} shots)")

    if args.relation == "auto":
        print(f"\n  All relation scores:")
        for name, code, score, _ in relation_results:
            marker = " <-- best" if score == best_score else ""
            kin_str = "KIN" if score >= threshold else "---"
            print(f"    {name:20s}  {score:.4f}  [{kin_str}]{marker}")

    if args.verbose:
        print(f"\n  Per-model scores (best relation: {best_name}):")
        for i, score in enumerate(best_per_model):
            print(f"    Model {i}: {score:.4f}")

    print("-" * 60)

    # -- Plot ----
    show_plot = not args.no_plot
    if show_plot or args.save_plot:
        save_path = None
        if args.save_plot:
            save_path = (
                args.save_plot
                if os.path.isabs(args.save_plot)
                else os.path.join(project_root, args.save_plot)
            )
        create_prediction_plot(
            img1_path,
            img2_path,
            relation_results,
            best_idx,
            threshold,
            n_models,
            save_path=save_path,
            show=show_plot,
        )


if __name__ == "__main__":
    main()
