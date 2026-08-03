# Quantum Kinship — Model Weights & Embedding Caches

This directory is organized into distinct subdirectories with **zero loose files** at the root:

```
weights/
├── active_ensemble/           # Primary deployment weights & metadata
│   ├── ensemble_kinship_full.pt      # Bundled 5-submodel ensemble (~25MB)
│   ├── ensemble_metadata.json        # Metadata and threshold settings
│   └── hybrid_kinship_improved_entangled.pt  # Best single fold model
├── folds/                     # Individual cross-validation fold models
│   ├── hybrid_kinship_improved_fold0_entangled.pt
│   ├── hybrid_kinship_improved_fold1_entangled.pt
│   ├── hybrid_kinship_improved_fold2_entangled.pt
│   ├── hybrid_kinship_improved_fold3_entangled.pt
│   └── hybrid_kinship_improved_fold4_entangled.pt
├── caches/                    # Pre-extracted FaceNet embedding (.pkl) files
│   ├── embeddings_cache.pkl          # KinFaceW & TSKinFace embeddings
│   ├── fiw_emb_cache.pkl             # FIW 73,814 face pairs embeddings
│   ├── fiw_retrain_emb_cache.pkl     # Merged FIW retraining cache
│   └── kinfacew-i_emb_cache.pkl      # KinFaceW-I face embeddings
└── archive/                   # Archived weights from legacy runs (v1–v3)
```

---

## ⚡ Embedding Caches Purpose
Pre-extracted 512-dimensional FaceNet embeddings skip CNN forward passes, enabling instant training and evaluation.
