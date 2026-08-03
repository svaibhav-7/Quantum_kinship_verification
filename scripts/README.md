# Quantum Kinship — Scripts Directory

This directory is organized into categorical subdirectories with **zero loose script files** at the root:

```
scripts/
├── training/                  # Model training & ensemble building scripts
│   ├── train_with_fiw.py            # Retrain ensemble on FIW 50% + existing data
│   ├── train_hybrid_improved.py     # Standard v4 cross-validation training pipeline
│   └── build_ensemble.py            # Package fold models into single ensemble .pt
├── evaluation/                # Benchmark evaluation & circuit validation scripts
│   ├── test_ensemble_on_unseen.py   # Evaluate ensemble on unseen FIW & KinFaceW-I
│   ├── evaluate_all_datasets.py     # Full multi-dataset evaluation suite
│   └── validate_quantum_circuit.py  # Qiskit AerSimulator timing & statevector verification
├── inference/                 # Prediction & live testing tools
│   ├── predict_user_images.py       # CLI tool for custom user face photo pairs
│   └── test_ensemble_live.py        # Live simulation test of deployed ensemble
└── archive/                   # Archived legacy scripts from earlier iterations (v1–v3)
```

---

## 💻 Common Commands

- **Retrain with FIW 50%**: `python scripts/training/train_with_fiw.py`
- **Test Unseen Data**: `python scripts/evaluation/test_ensemble_on_unseen.py --max-fiw-pairs 0`
- **Predict User Photos**: `python scripts/inference/predict_user_images.py --img1 photo1.jpg --img2 photo2.jpg --relation fd`
