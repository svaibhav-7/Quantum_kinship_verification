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

## Experimental VQC image prediction

The VQC experiment has no pretrained deployment checkpoint. First train the
dual-readout model (joint amplitude encoding, a trainable simulated circuit,
fidelity plus per-qubit expectation measurements) from the project's
FaceNet embedding cache:

```powershell
python scripts/training/train_amplitude_vqc.py --dataset KinFaceW-I
```

Then score two face-cropped photos:

```powershell
python scripts/inference/predict_vqc.py --img1 parent.jpg --img2 child.jpg
```

Training saves `weights/deploy/amplitude_vqc.pt`. The training and prediction
scripts use FaceNet embeddings without face detection, matching the cached
training preprocessing; provide tightly cropped face images. The VQC is
simulated in PyTorch and is an experimental research model, not a quantum
hardware deployment. Its validation metrics are not a guarantee for any
individual pair.

## RFIW test-pair coverage audit

Check whether the locally available official test pairs differ by kin label
or relation. The audit uses the project's RFIW embedding cache and writes
JSON and Markdown summaries under `results/honest/`:

```powershell
python scripts/evaluation/audit_rfiw_test_coverage.py
```

Its two-endpoint family-bootstrap interval is descriptive; it is not a
hypothesis-test p-value.
