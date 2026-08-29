# -*- coding: utf-8 -*-
"""
MASTER MODEL COMPARISON RUNNER

Executes all 12 modules for the 4 candidate models sequentially, then
generates the final best-model identification summary (JSON + markdown).

Usage:
    python research_experiments/master_model_comparison/run_all.py
"""

import os
import sys
import json
import time
import importlib

_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)
_research_root = os.path.dirname(_current)
_project_root = os.path.dirname(_research_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def run_module(mod_name, func_name):
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn()


def main():
    t_start = time.time()
    print("=" * 72)
    print("  MASTER MODEL COMPARISON - 12 MODULES x 4 MODELS")
    print("=" * 72)

    modules = [
        ("research_experiments.master_model_comparison.01_ablation_study.run_ablation", "run_ablation"),
        ("research_experiments.master_model_comparison.02_sota_literature_comparison.run_sota", "run_sota"),
        ("research_experiments.master_model_comparison.03_statistical_significance.run_statistics", "run_statistics"),
        ("research_experiments.master_model_comparison.04_explainability_tsne.run_tsne", "run_tsne"),
        ("research_experiments.master_model_comparison.05_roc_pr_curves.run_roc_pr", "run_roc_pr"),
        ("research_experiments.master_model_comparison.06_threshold_analysis.run_threshold", "run_threshold"),
        ("research_experiments.master_model_comparison.07_robustness_degradation.run_robustness", "run_robustness"),
        ("research_experiments.master_model_comparison.08_computational_efficiency.run_efficiency", "run_efficiency"),
        ("research_experiments.master_model_comparison.09_qualitative_error_analysis.run_error_analysis", "run_error_analysis"),
        ("research_experiments.master_model_comparison.10_cross_dataset.run_cross_dataset", "run_cross_dataset"),
        ("research_experiments.master_model_comparison.11_baseline_comparisons.run_baseline", "run_baseline"),
        ("research_experiments.master_model_comparison.12_ensemble_weight_ablation.run_weight_ablation", "run_weight_ablation"),
    ]

    results_store = {}
    for mod_path, func in modules:
        try:
            print(f"\n" + "=" * 72)
            print(f"  RUNNING: {mod_path}.{func}")
            print("=" * 72)
            res = run_module(mod_path, func)
            name = mod_path.split(".")[-1]
            results_store[name] = res
        except Exception as e:
            import traceback
            print(f"  [ERROR] Module {mod_path} failed: {e}")
            traceback.print_exc()

    # Generate best-model summary
    _generate_summary(results_store, _project_root, _current)

    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"  MASTER MODEL COMPARISON COMPLETE in {elapsed:.1f}s")
    print("=" * 72)


def _generate_summary(results, project_root, current_dir):
    """Aggregate per-model metrics across modules and rank models."""
    summary = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "models": {}, "ranking": [], "best_model": None}

    # Performance metrics per model from module 05 (ROC/PR) & module 10 (cross-dataset),
    # and module 11 (baseline on FIW).
    model_scores = {
        k: {"accuracy_mean": 0.0, "roc_auc_mean": 0.0, "n": 0, "inputs": 0}
        for k in ["ensemble_kinship_full", "meta_ensemble_kinship", "ensemble_kinship_fiw", "best_checkpoint"]
    }

    # Module 05: per-model ROC-AUC across 4 datasets
    if "run_roc_pr" in results:
        for mkey in model_scores:
            if mkey in results["run_roc_pr"]:
                for dname, met in results["run_roc_pr"][mkey].items():
                    model_scores[mkey]["roc_auc_mean"] += met.get("roc_auc", 0)
                    model_scores[mkey]["n"] += 1
                if model_scores[mkey]["n"] > 0:
                    model_scores[mkey]["roc_auc_mean"] /= model_scores[mkey]["n"]

    # Module 10: cross-dataset accuracy (mean over 4 test datasets)
    if "run_cross_dataset" in results:
        for mkey in model_scores:
            if mkey in results["run_cross_dataset"]:
                vals = list(results["run_cross_dataset"][mkey].values())
                model_scores[mkey]["accuracy_mean"] += np_mean(vals) * 0.5
                model_scores[mkey]["inputs"] += 4

    # Module 11: baseline (FIW accuracy) - use as additional accuracy signal
    if "run_baseline" in results:
        for mkey in model_scores:
            if mkey in results["run_baseline"]:
                model_scores[mkey]["accuracy_mean"] += results["run_baseline"][mkey].get("accuracy", 0) * 0.5
                model_scores[mkey]["inputs"] += 1

    # Module 06: threshold (FIW max accuracy)
    acc_thresh = {}
    if "run_threshold" in results:
        for mkey in model_scores:
            if mkey in results["run_threshold"]:
                acc_thresh[mkey] = results["run_threshold"][mkey].get("max_accuracy", 0)

    # Module 03: statistical (FIW accuracy)
    if "run_statistics" in results:
        for mkey in model_scores:
            if mkey in results["run_statistics"]:
                model_scores[mkey]["accuracy_mean"] += results["run_statistics"][mkey].get("accuracy", 0) * 0.5
                model_scores[mkey]["inputs"] += 1

    # Normalize accuracy_mean by number of contributing inputs
    for k in model_scores:
        if model_scores[k]["inputs"] > 0:
            model_scores[k]["accuracy_mean"] /= model_scores[k]["inputs"]
        if acc_thresh.get(k) is not None:
            model_scores[k]["accuracy_mean"] = (model_scores[k]["accuracy_mean"] + acc_thresh[k]) / 2

    # Build summary entries
    for mkey, sc in model_scores.items():
        summary["models"][mkey] = {
            "mean_accuracy": round(float(sc["accuracy_mean"]), 2),
            "mean_roc_auc": round(float(sc["roc_auc_mean"]), 4),
        }

    # Rank by weighted aggregate (accuracy + roc_auc)
    ranked = sorted(model_scores.keys(), key=lambda k: (
        summary["models"][k]["mean_accuracy"] + summary["models"][k]["mean_roc_auc"] * 100
    ), reverse=True)

    summary["ranking"] = ranked
    summary["best_model"] = ranked[0] if ranked else None
    summary["best_model_label"] = {
        "ensemble_kinship_full": "Base Ensemble (5 folds)",
        "meta_ensemble_kinship": "Meta-Ensemble (11 models)",
        "ensemble_kinship_fiw": "FIW Ensemble (5 folds)",
        "best_checkpoint": "Best Checkpoint (1 model)",
    }.get(summary["best_model"], "")

    json_path = os.path.join(current_dir, "best_model_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write a markdown report
    md_path = os.path.join(current_dir, "MASTER_MODEL_COMPARISON_REPORT.md")
    with open(md_path, "w") as f:
        f.write("# Quantum Kinship - Master Model Comparison Report\n\n")
        f.write(f"Generated: {summary['generated_at']}\n\n")
        f.write("## Best Performing Model\n\n")
        f.write(f"**{summary.get('best_model_label','N/A')}** (`{summary.get('best_model','N/A')}`)\n\n")
        f.write("## Model Rankings\n\n")
        f.write("| Rank | Model | Mean Accuracy (%) | Mean ROC-AUC |\n")
        f.write("|------|-------|-------------------|---------------|\n")
        for i, k in enumerate(ranked, 1):
            f.write(f"| {i} | {summary['models'][k]['mean_accuracy']} "
                    f"| {summary['models'][k]['mean_roc_auc']} |\n")
        f.write("\n## Summary of All 12 Modules\n\n")
        f.write("- 01: Ablation Study (per-model JSON + plots)\n")
        f.write("- 02: SOTA Literature Comparison (JSON + plot)\n")
        f.write("- 03: Statistical Significance (JSON + plot)\n")
        f.write("- 04: Explainability t-SNE (PNG per model)\n")
        f.write("- 05: ROC/PR Curves (JSON + plots)\n")
        f.write("- 06: Threshold Analysis (JSON + plot)\n")
        f.write("- 07: Robustness/Degradation (JSON + plots)\n")
        f.write("- 08: Computational Efficiency (JSON + plot)\n")
        f.write("- 09: Error Analysis (JSON + plot)\n")
        f.write("- 10: Cross-Dataset Matrix (JSON + heatmaps)\n")
        f.write("- 11: Baseline Comparisons (JSON + plot)\n")
        f.write("- 12: Ensemble Weight Ablation (JSON + plot)\n")

    print(f"\n[SUMMARY] best model: {summary.get('best_model')}")
    print(f"[SUMMARY] saved to {json_path}")


def np_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


if __name__ == "__main__":
    main()

