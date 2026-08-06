# -*- coding: utf-8 -*-
"""
MASTER RESEARCH RUNNER — SEQUENTIAL EXECUTION OF ALL 12 PRIORITY MODULES
Executes Module 1 through Module 12 sequentially, aggregates outputs, copies figures to artifacts,
and compiles the final PDF and Markdown master report.
"""

import os
import sys
import shutil
import time
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
brain_dir = r"C:\Users\svrao\.gemini\antigravity-ide\brain\a7cfb6c9-bc14-475d-823d-b240d3fe6363"

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

    # Copy all PNG figures to brain artifact directory
    outputs_dir = os.path.join(current_dir, "outputs")
    os.makedirs(brain_dir, exist_ok=True)
    png_count = 0
    for root, _, files in os.walk(outputs_dir):
        for f in files:
            if f.endswith(".png"):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(brain_dir, f)
                shutil.copy(src_path, dst_path)
                png_count += 1

    print(f"\n[COPIED {png_count} PNG FIGURES TO ARTIFACT DIRECTORY]")

    # Generate Master PDF & Markdown Report
    mod_rep = importlib.import_module("research_experiments.master_report_generator")
    mod_rep.generate_pdf_report()

    t_end = time.time()
    print("\n=================================================================")
    print(f" [SUCCESS] ALL 12 PRIORITY MODULES EXECUTED IN {t_end - t_start:.2f} SECONDS!")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
