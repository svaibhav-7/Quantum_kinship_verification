# -*- coding: utf-8 -*-
"""
MASTER RESEARCH RUNNER — SEQUENTIAL EXECUTION OF ALL 12 PRIORITY MODULES
Executes Module 1 through Module 12 sequentially, aggregates outputs.
NOTE: This version has been corrected to remove hard-coded paths and focus on local reproducibility.
"""

import os
import sys
import time
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Helper function to dynamically import numbered subfolders
def run_module_by_path(module_rel_path, function_name):
    mod = importlib.import_module(module_rel_path)
    fn = getattr(mod, function_name)
    return fn()

def main():
    t_start = time.time()
    print("=================================================================")
    print("  EXECUTING MASTER RESEARCH SUITE — ALL 12 PRIORITY MODULES")
    print("=================================================================")

    # Module 1
    run_module_by_path("research_experiments.01_ablation_study.run_ablation_study", "run_ablation")

    # Module 2
    run_module_by_path("research_experiments.02_sota_literature_comparison.run_sota_comparison", "run_sota_comparison")

    # Module 3
    run_module_by_path("research_experiments.03_statistical_significance.run_statistical_tests", "run_statistical_tests")

    # Module 4
    run_module_by_path("research_experiments.04_explainability_tsne.run_explainability", "run_explainability")

    # Module 5
    run_module_by_path("research_experiments.05_roc_pr_curves.run_roc_pr_curves", "run_roc_pr_curves")

    # Module 6
    run_module_by_path("research_experiments.06_threshold_analysis.run_threshold_analysis", "run_threshold_analysis")

    # Module 7
    run_module_by_path("research_experiments.07_robustness_degradation.run_robustness_tests", "run_robustness_tests")

    # Module 8
    run_module_by_path("research_experiments.08_computational_efficiency.run_efficiency_analysis", "run_efficiency_analysis")

    # Module 9
    run_module_by_path("research_experiments.09_qualitative_error_analysis.run_error_analysis", "run_error_analysis")

    # Module 10
    run_module_by_path("research_experiments.10_cross_dataset.run_cross_dataset", "run_cross_dataset")

    # Module 11
    run_module_by_path("research_experiments.11_baseline_comparisons.run_baseline_comparison", "run_baseline_comparison")

    # Module 12
    run_module_by_path("research_experiments.12_ensemble_weight_ablation.run_weight_ablation", "run_weight_ablation")

    # Count output files for reporting
    outputs_dir = os.path.join(current_dir, "outputs")
    png_count = 0
    json_count = 0
    if os.path.exists(outputs_dir):
        for root, _, files in os.walk(outputs_dir):
            for f in files:
                if f.endswith(".png"):
                    png_count += 1
                elif f.endswith(".json"):
                    json_count += 1

    print(f"\n[OUTPUT STATS] {png_count} PNG figures, {json_count} JSON results in {outputs_dir}")

    # Generate Master PDF & Markdown Report
    try:
        mod_rep = importlib.import_module("research_experiments.master_report_generator")
        mod_rep.generate_pdf_report()
    except Exception as e:
        print(f"  [WARNING] Could not generate master report: {e}")

    t_end = time.time()
    print("\n=================================================================")
    print(f" [SUCCESS] ALL 12 PRIORITY MODULES EXECUTED IN {t_end - t_start:.2f} SECONDS!")
    print("=================================================================")
    print("")
    print("NEXT STEPS FOR REPRODUCIBLE RESEARCH:")
    print("1. Verify all outputs are in research_experiments/outputs/")
    print("2. Check master_research_summary.md for limitations and next steps")
    print("3. Address known issues before attempting publication:")
    print("   - FIW dataset contamination (same-person pairs)")
    print("   - Data leakage in training/test splits")
    print("   - FaceNet preprocessing mismatches")
    print("   - TSKinFace label errors")
    print("   - Need for retraining with corrected pipelines")
    print("")

if __name__ == "__main__":
    main()