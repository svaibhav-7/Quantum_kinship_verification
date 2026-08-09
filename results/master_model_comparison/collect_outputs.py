# -*- coding: utf-8 -*-
"""
MASTER OUTPUT COLLECTOR

Consolidates all 12 module outputs (JSON + plots) for the 4 models into a
single master folder `master_outputs/` with organized subfolders:
  - master_outputs/json/   -> all per-model + summary JSON (prefixed by module)
  - master_outputs/plots/  -> all generated plots (prefixed by module)

Also writes:
  - master_outputs/master_comparison_summary.json -> aggregated ranking
  - master_outputs/MASTER_MODEL_COMPARISON_REPORT.md -> human-readable report

Usage:
    python research_experiments/master_model_comparison/collect_outputs.py
"""

import os
import sys
import json
import shutil

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = _current

MODULE_DIRS = {
    "01_ablation_study": "01_ablation_study",
    "02_sota_literature_comparison": "02_sota_literature_comparison",
    "03_statistical_significance": "03_statistical_significance",
    "04_explainability_tsne": "04_explainability_tsne",
    "05_roc_pr_curves": "05_roc_pr_curves",
    "06_threshold_analysis": "06_threshold_analysis",
    "07_robustness_degradation": "07_robustness_degradation",
    "08_computational_efficiency": "08_computational_efficiency",
    "09_qualitative_error_analysis": "09_qualitative_error_analysis",
    "10_cross_dataset": "10_cross_dataset",
    "11_baseline_comparisons": "11_baseline_comparisons",
    "12_ensemble_weight_ablation": "12_ensemble_weight_ablation",
}

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


def sanitize_name(name):
    """Make a filename-safe string from a module/model/plot name."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")


def collect():
    master = os.path.join(_module_root, "master_outputs")
    json_dir = os.path.join(master, "json")
    plots_dir = os.path.join(master, "plots")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    collected_json = []
    collected_plots = []

    for module_prefix, module_dir_name in MODULE_DIRS.items():
        src_dir = os.path.join(_module_root, module_dir_name)
        if not os.path.isdir(src_dir):
            continue
        for fname in sorted(os.listdir(src_dir)):
            fpath = os.path.join(src_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if fname.endswith(".json"):
                dst = os.path.join(json_dir, f"{module_prefix}__{fname}")
                shutil.copy2(fpath, dst)
                collected_json.append(dst)
            elif fname.endswith(".png"):
                dst = os.path.join(plots_dir, f"{module_prefix}__{fname}")
                shutil.copy2(fpath, dst)
                collected_plots.append(dst)

    # Build aggregated summary from collected JSONs
    summary = aggregate_summary(json_dir, collected_json)

    summary_path = os.path.join(master, "master_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write markdown report
    md_path = os.path.join(master, "MASTER_MODEL_COMPARISON_REPORT.md")
    with open(md_path, "w") as f:
        write_report(f, summary, collected_json, collected_plots)

    print(f"[COLLECT] JSON files: {len(collected_json)}")
    print(f"[COLLECT] Plot files: {len(collected_plots)}")
    print(f"[COLLECT] Summary -> {summary_path}")
    print(f"[COLLECT] Report  -> {md_path}")
    return summary


def aggregate_summary(json_dir, collected_json):
    """Aggregate per-model metrics across modules and rank models."""
    acc_per_model = {k: [] for k in MODEL_KEYS}
    auc_per_model = {k: [] for k in MODEL_KEYS}

    # Robustness JSONs contain arrays of accuracies per noise/severity -> weight less
    ROBUSTNESS_KEYS = {"Gaussian Noise", "Gaussian Blur", "JPEG Compression",
                       "Random Occlusion", "Brightness Shifting", "Rotation / Tilt"}
    # Threshold JSONs contain threshold_curve with accuracy arrays
    THRESHOLD_KEYS = {"threshold_curve", "max_accuracy", "optimal_threshold"}

    for jpath in collected_json:
        base = os.path.basename(jpath)
        try:
            with open(jpath) as f:
                data = json.load(f)
        except Exception:
            continue

        # Determine which model this JSON belongs to
        model_key = None
        for k in MODEL_KEYS:
            if k in base or (isinstance(data, dict) and data.get("model") == k):
                model_key = k
                break
        if model_key is None:
            continue

        if not isinstance(data, dict):
            continue

        # -- Module 12 baselines: accuracy + roc_auc --
        if "accuracy" in data and isinstance(data["accuracy"], (int, float)) and \
           "roc_auc" in data and isinstance(data["roc_auc"], (int, float)):
            acc_per_model[model_key].append(float(data["accuracy"]))
            auc_per_model[model_key].append(float(data["roc_auc"]))
            continue

        # -- Module 05 ROC/PR: {dataset: {roc_auc, pr_auc}} --
        if all(isinstance(v, dict) and "roc_auc" in v for v in data.values() if isinstance(v, dict)) and data:
            ds_aucs = [float(v["roc_auc"]) for v in data.values() if isinstance(v, dict) and "roc_auc" in v]
            if ds_aucs:
                auc_per_model[model_key].append(float(sum(ds_aucs) / len(ds_aucs)))
                continue

        # -- Module 06 Threshold: max_accuracy as single accuracy signal --
        if "max_accuracy" in data and isinstance(data["max_accuracy"], (int, float)):
            acc_per_model[model_key].append(float(data["max_accuracy"]))
            continue

        # -- Module 10 Cross-dataset / ablation: {dataset: accuracy} -> mean --
        _skip = ("model", "module", "total_test_pairs", "true_positives",
                 "true_negatives", "false_positives", "false_negatives",
                 "n_samples", "roc_auc", "pr_auc", "f1_score", "kin_mean", "nonkin_mean")
        ds_accs = [float(v) for dk, v in data.items()
                   if dk not in _skip and isinstance(v, (int, float)) and v < 1000]
        if len(ds_accs) >= 3 and not any(k in data for k in THRESHOLD_KEYS) and \
           not any(k in data for k in ROBUSTNESS_KEYS):
            acc_per_model[model_key].append(float(sum(ds_accs) / len(ds_accs)))
            continue

        # -- Module 03 Statistical: accuracy + bootstrap CI --
        if "accuracy" in data and isinstance(data["accuracy"], (int, float)):
            acc_per_model[model_key].append(float(data["accuracy"]))
            continue

        # -- Module 09 Error analysis: compute accuracy from confusion counts --
        if all(k in data for k in ("true_positives", "true_negatives", "false_positives", "false_negatives")):
            tp = float(data["true_positives"]); tn = float(data["true_negatives"])
            fp = float(data["false_positives"]); fn = float(data["false_negatives"])
            total = tp + tn + fp + fn
            if total > 0:
                acc_per_model[model_key].append(float((tp + tn) / total * 100))
            continue

        # -- Module 08 Efficiency: no accuracy, skip --
        if "parameters" in data:
            continue

    model_entries = {}
    for k in MODEL_KEYS:
        accs = acc_per_model[k]
        aucs = auc_per_model[k]
        model_entries[k] = {
            "label": MODEL_LABELS[k],
            "mean_accuracy": round(float(sum(accs) / len(accs)), 2) if accs else None,
            "mean_roc_auc": round(float(sum(aucs) / len(aucs)), 4) if aucs else None,
            "n_accuracy_signals": len(accs),
            "n_auc_signals": len(aucs),
        }

    # Rank by combined score (accuracy + auc*100), falling back to available
    def score(k):
        e = model_entries[k]
        s = 0.0
        if e["mean_accuracy"] is not None:
            s += e["mean_accuracy"]
        if e["mean_roc_auc"] is not None:
            s += e["mean_roc_auc"] * 100
        return s

    ranked = sorted(MODEL_KEYS, key=score, reverse=True)
    best = ranked[0] if ranked else None

    return {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "models": model_entries,
        "ranking": ranked,
        "best_model": best,
        "best_model_label": MODEL_LABELS.get(best, ""),
    }


def write_report(f, summary, collected_json, collected_plots):
    f.write("# Quantum Kinship - Master Model Comparison Report\n\n")
    f.write(f"Generated: {summary['generated_at']}\n\n")
    f.write("## Best Performing Model\n\n")
    f.write(f"**{summary['best_model_label']}** (`{summary['best_model']}`)\n\n")
    f.write("## Model Rankings\n\n")
    f.write("| Rank | Model | Mean Accuracy (%) | Mean ROC-AUC |\n")
    f.write("|------|-------|-------------------|---------------|\n")
    for i, k in enumerate(summary["ranking"], 1):
        e = summary["models"][k]
        acc = f"{e['mean_accuracy']}" if e["mean_accuracy"] is not None else "N/A"
        auc = f"{e['mean_roc_auc']}" if e["mean_roc_auc"] is not None else "N/A"
        f.write(f"| {i} | {e['label']} | {acc} | {auc} |\n")

    f.write("\n## Collected JSON Files\n\n")
    for j in collected_json:
        f.write(f"- `{os.path.basename(j)}`\n")
    f.write("\n## Collected Plot Files\n\n")
    for p in collected_plots:
        f.write(f"- `{os.path.basename(p)}`\n")


if __name__ == "__main__":
    collect()
