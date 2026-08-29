# Large-Scale Generalization Evaluation (4 Models x 4 Unseen Datasets)

Generated: 2026-08-09 14:38:31
Datasets evaluated: FIW, KinFaceW-I, KinFaceW-II, TSKinFace

## Best Model on Unseen General Data

**Meta-Ensemble (11 models)** (`meta_ensemble_kinship`)

## Generalization Ranking (mean across unseen datasets)

| Rank | Model | Mean Accuracy (%) | Mean ROC-AUC | Mean F1 (%) |
|------|-------|-------------------|--------------|-------------|
| 1 | Meta-Ensemble (11 models) | 72.19 | 0.7971 | 76.04 |
| 2 | ensemble_kinship_full | 70.87 | 0.7838 | 74.18 |
| 3 | ensemble_kinship_fiw | 67.71 | 0.7362 | 71.88 |
| 4 | best_checkpoint | 64.21 | 0.6691 | 69.89 |

## Per-Dataset Accuracy (%)

| Model | FIW | KinFaceW-I | KinFaceW-II | TSKinFace |
|-------|------|------|------|------|
| meta_ensemble_kinship | 76.0 | 67.4 | 68.2 | 77.2 |
| ensemble_kinship_full | 60.2 | 70.5 | 69.5 | 83.3 |
| ensemble_kinship_fiw | 72.8 | 61.6 | 62.9 | 73.5 |
| best_checkpoint | 77.8 | 56.5 | 57.8 | 64.8 |

## Per-Dataset ROC-AUC

| Model | FIW | KinFaceW-I | KinFaceW-II | TSKinFace |
|-------|------|------|------|------|
| meta_ensemble_kinship | 0.8596 | 0.7365 | 0.7525 | 0.8398 |
| ensemble_kinship_full | 0.6919 | 0.7672 | 0.7740 | 0.9021 |
| ensemble_kinship_fiw | 0.8175 | 0.6752 | 0.6891 | 0.7631 |
| best_checkpoint | 0.8636 | 0.5923 | 0.6160 | 0.6044 |

## Output Files

Plots generated in `master_outputs/generalization_eval/`:
- `accuracy_4models.png`
- `best_on_unseen_ranking.png`
- `confusion_FIW_4models.png`
- `distributions_FIW_4models.png`
- `f1_4models.png`
- `pr_FIW_combined_4models.png`
- `precision_4models.png`
- `recall_4models.png`
- `roc_FIW_4models.png`
- `roc_FIW_combined_4models.png`
- `roc_KinFaceW-II_4models.png`
- `roc_KinFaceW-I_4models.png`
- `roc_TSKinFace_4models.png`
- `roc_auc_4models.png`
- `threshold_FIW_4models.png`