# MASTER MODEL COMPARISON - 12 MODULE TASKS

Goal: Create all 12 research components for each of the 4 models, produce JSON + plots per model, store in master folder, and identify best-performing model.

## Models (4)
1. `ensemble_kinship_full.pt` - Base Ensemble (5 folds)
2. `meta_ensemble_kinship.pt` - Meta-Ensemble (11 models hierarchical fusion)
3. `ensemble_kinship_fiw.pt` - FIW Ensemble (5 folds)
4. `best_checkpoint.pt` - Fine-tuned single checkpoint

## Structure
- `research_experiments/master_model_comparison/`
  - `common_data.py` - shared dataset + embedding cache loading
  - `common_models.py` - shared model registry/loader
  - `01_ablation_study/` ... `12_ensemble_weight_ablation/`
  - `run_all.py` - master runner
  - `best_model_summary.json` - final best-model identification
  - `master_outputs/` - consolidated JSON + plots master folder

## Task Plan
- [x] 1. Analyze repo & understand models/data
- [x] 2. Build plan & get approval
- [x] 3. Create TODO.md
- [x] 4. Create common_data.py (datasets + emb cache)
- [x] 5. Create common_models.py (load 4 models)
- [x] 6. Module 01 - Ablation Study (per model)
- [x] 7. Module 02 - SOTA Literature Comparison (per model)
- [x] 8. Module 03 - Statistical Significance (per model)
- [x] 9. Module 04 - Explainability t-SNE (per model)
- [x] 10. Module 05 - ROC/PR Curves (per model)
- [x] 11. Module 06 - Threshold Analysis (per model)
- [x] 12. Module 07 - Robustness/Degradation (per model)
- [x] 13. Module 08 - Computational Efficiency (per model)
- [x] 14. Module 09 - Qualitative Error Analysis (per model)
- [x] 15. Module 10 - Cross-Dataset Matrix (per model)
- [x] 16. Module 11 - Baseline Comparisons (per model)
- [x] 17. Module 12 - Ensemble Weight Ablation (per model)
- [x] 18. Create run_all.py master runner
- [x] 19. Generate best_model_summary.json + markdown
- [x] 20. Fix missing plots (module 03, 04, 12)
- [x] 21. Create collect_outputs.py to consolidate into master_outputs/
- [x] 22. Execute & verify all outputs (12 modules x 4 models JSON + plots)
- [x] 23. Identify best-performing model from aggregated results -> meta_ensemble_kinship (Mean Acc 75.63%, ROC-AUC 0.8388)
