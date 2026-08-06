import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_metric_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "ensemble_test_results" in data:
        metrics = data["ensemble_test_results"]
    else:
        metrics = data

    return {
        "accuracy": float(metrics.get("accuracy_optimal", metrics.get("accuracy", 0.0))),
        "roc_auc": float(metrics.get("roc_auc", 0.0)),
    }


def build_comparison_plot(output_path=None):
    project_root = ROOT
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    active_path = results_dir / "training_metrics" / "ensemble_live_evaluation.json"
    retrained_path = results_dir / "fiw_retrained_metrics" / "fiw_retrained_results.json"

    if not active_path.exists():
        raise FileNotFoundError(f"Missing active ensemble metrics: {active_path}")
    if not retrained_path.exists():
        raise FileNotFoundError(f"Missing FIW retrained metrics: {retrained_path}")

    active_metrics = load_metric_summary(active_path)
    retrained_metrics = load_metric_summary(retrained_path)

    labels = ["Active Ensemble", "FIW Retrained"]
    accuracies = [active_metrics["accuracy"], retrained_metrics["accuracy"]]
    auc_scores = [active_metrics["roc_auc"], retrained_metrics["roc_auc"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    fig.suptitle("Model Comparison — Active Ensemble vs FIW Retrained", fontsize=14, fontweight="bold")

    bars1 = axes[0].bar(labels, accuracies, color=["#4A90E2", "#E2849A"], edgecolor="black", linewidth=0.7)
    axes[0].set_title("Accuracy (optimal threshold)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)

    bars2 = axes[1].bar(labels, auc_scores, color=["#4A90E2", "#E2849A"], edgecolor="black", linewidth=0.7)
    axes[1].set_title("ROC-AUC")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars2, auc_scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is None:
        output_path = results_dir / "model_comparison_summary.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


if __name__ == "__main__":
    output = build_comparison_plot()
    print(f"Saved comparison plot to: {output}")
