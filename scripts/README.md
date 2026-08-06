# Quantum Kinship — Scripts Directory

This directory is organized into categorical subdirectories with **zero loose script files** at the root:

```
scripts/
├── training/                  # Model training & ensemble building scripts
│   ├── train_with_fiw.py            # Retrain ensemble on FIW 50% + existing data
│   ├── train_hybrid_improved.py     # Standard cross-validation training pipeline
│   ├── build_ensemble.py            # Package fold models into single 5-fold ensemble .pt
│   └── build_meta_ensemble.py       # Build 11-model hierarchical domain meta-ensemble
├── evaluation/                # Benchmark evaluation & plotting scripts
│   ├── generate_meta_ensemble_plots_and_json.py # Generate ROC, accuracy & breakdown plots
│   ├── test_ensemble_on_unseen.py   # Evaluate ensemble on unseen FIW & KinFaceW-I
│   ├── evaluate_all_datasets.py     # Full multi-dataset evaluation suite
│   └── validate_quantum_circuit.py  # Qiskit AerSimulator timing & statevector verification
├── inference/                 # Prediction & live testing tools
│   ├── predict_user_images.py       # CLI tool for custom user face photo pairs
│   └── test_ensemble_live.py        # Live simulation test of deployed ensemble
└── archive/                   # Archived legacy scripts from earlier iterations
```

---

## 💻 Common Commands

- **Build Meta-Ensemble**: `python scripts/training/build_meta_ensemble.py`
- **Generate Meta-Ensemble Metrics & Plots**: `python scripts/evaluation/generate_meta_ensemble_plots_and_json.py`
- **Test Unseen Data**: `python scripts/evaluation/test_ensemble_on_unseen.py --max-fiw-pairs 0`
- **Predict User Photos**: `python scripts/inference/predict_user_images.py --img1 photo1.jpg --img2 photo2.jpg --relation fd`
