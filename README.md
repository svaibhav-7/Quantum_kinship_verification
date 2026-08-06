# Quantum Kinship Verification System

An advanced quantum-inspired facial kinship verification framework leveraging entangled state vector representations (`n_qubits=8`), relation-conditioned attention projection, and a **Hierarchical Meta-Ensemble Classifier** for cross-domain generalization.

---

## 📁 Repository Structure

```
Quantum_kinship/
├── src/                          # Core Python source packages
│   ├── models_improved.py        # HybridKinshipClassifier & MetaEnsembleKinshipClassifier
│   ├── quantum_core.py           # Quantum circuit & simulator components
│   └── data_loaders.py           # Pair loading & embedding cache utilities
├── scripts/                      # Categorical execution scripts
│   ├── training/                 # Model training & meta-ensemble building scripts
│   │   ├── build_meta_ensemble.py     # Bundles 11 sub-models into meta_ensemble_kinship.pt
│   │   ├── train_with_fiw.py        # FIW dataset retraining pipeline
│   │   └── build_ensemble.py        # Package fold models into 5-fold ensemble
│   ├── evaluation/               # Metric evaluation & plot generation scripts
│   │   ├── generate_meta_ensemble_plots_and_json.py # Comprehensive plot generator
│   │   ├── test_ensemble_on_unseen.py # Benchmark evaluation suite
│   │   └── validate_quantum_circuit.py# Circuit timing & simulation check
│   ├── inference/                # Real-time prediction scripts
│   │   ├── predict_user_images.py   # CLI tool for custom user face pairs
│   │   └── test_ensemble_live.py    # Simulated live API test
│   └── archive/                  # Legacy training iterations (v1–v3)
├── weights/                      # Model weights & embedding caches
│   ├── active_ensemble/          # Meta-Ensemble checkpoint (meta_ensemble_kinship.pt)
│   ├── secondary_active_ensemble/# FIW 5-fold ensemble checkpoint
│   └── caches/                   # Pre-computed FaceNet embedding caches
├── results/                      # Evaluation artifacts, plots & reports
│   ├── plots/                    # High-res ROC curves, metric comparisons & per-relation charts
│   ├── reports/                  # PDF reports & comprehensive evaluation metrics JSON
│   └── meta_ensemble_comprehensive_metrics.json
├── tests/                        # Unit tests & verification scripts
│   ├── test_fix.py               # Physics regularization test suite
│   └── test_family_split.py      # FIW family split verification
├── docs/                         # Architecture documentation & deep-dives
├── paper/                        # Academic paper LaTeX source code
└── main.py                       # Project entry point & quick demo
```

---

## 🚀 Quick Start & Common Commands

### 1. Run Demo Prediction
```bash
python main.py
```

### 2. Build Meta-Ensemble Model (11 Sub-Models)
```bash
python scripts/training/build_meta_ensemble.py
```

### 3. Generate Comprehensive Metrics & Plots
```bash
python scripts/evaluation/generate_meta_ensemble_plots_and_json.py
```

### 4. Run Custom Face Pair Prediction
```bash
python scripts/inference/predict_user_images.py --img1 photo1.jpg --img2 photo2.jpg --relation fd
```

### 5. Run Unit Tests
```bash
python -m unittest discover -s tests
```
