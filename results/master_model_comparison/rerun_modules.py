# -*- coding: utf-8 -*-
"""
RE-RUN HELPER for the 3 failed modules (03, 04, 12) after bug fixes.
Only re-runs these modules (not the full pipeline) to save compute.
"""
import os
import sys
import time

_current = os.path.dirname(os.path.abspath(__file__))
if _current not in sys.path:
    sys.path.insert(0, _current)
_research_root = os.path.dirname(_current)
_project_root = os.path.dirname(_research_root)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

modules = [
    ("research_experiments.master_model_comparison.03_statistical_significance.run_statistics", "run_statistics"),
    ("research_experiments.master_model_comparison.04_explainability_tsne.run_tsne", "run_tsne"),
    ("research_experiments.master_model_comparison.12_ensemble_weight_ablation.run_weight_ablation", "run_weight_ablation"),
]

if __name__ == "__main__":
    for mod_path, func in modules:
        t0 = time.time()
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, func)
            res = fn()
            print(f"[OK] {mod_path}.{func} done in {time.time()-t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"[ERROR] {mod_path}.{func} failed: {e}")
            traceback.print_exc()

