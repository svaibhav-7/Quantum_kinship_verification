# Facial Kinship Verification: Leakage-Free Evaluation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A leakage-free evaluation pipeline for facial kinship verification, with a
> compact classifier trained under it and a tested negative result for a
> quantum-inspired similarity metric.

---

## Overview

Facial kinship verification decides whether two people are biologically
related from unconstrained photographs. This repository contains a
**leakage-free evaluation pipeline** for the task, plus a compact classifier
trained under it.

The headline contribution is methodological. While building this we found
that the field's standard protocols make results look much better than they
are:

- **Identity leakage.** Pair-level random splits put an identity from
  **100% of test pairs** (11,424/11,424) into the training set.
- **A dataset shortcut.** In TSKinFace every positive is same-family and every
  negative is cross-family (800/800 vs 0/800), so a model can score ~84% by
  recognising a shared photo session rather than kinship. Closing it costs
  about **9 ROC-AUC points**.
- **A negative result.** A quantum-inspired SWAP-test fidelity module adds
  **nothing measurable** over standard pair features (20 paired folds,
  p = 0.158 accuracy, p = 0.982 AUC) while costing 6-14x runtime.

Under corrected grouped 5-fold evaluation, a 1.9M-parameter classifier reaches
**75.66% mean accuracy / 0.8497 mean ROC-AUC** across four datasets.

Full methodology, per-fold numbers and audit results: **[RESULTS_HONEST.md](RESULTS_HONEST.md)**.

---

## Method

1. **Frozen FaceNet embeddings** (InceptionResnetV1, VGGFace2) -- 512-d per face.
2. **Shared Siamese encoder** so both faces are treated identically; kinship is
   symmetric, so only symmetric pair features are used
   (`sum`, `|difference|`, `product`, `cosine`).
3. **Relation conditioning** on FD / FS / MD / MS.
4. **Quantum-inspired branch** (optional): embeddings are projected to 8
   rotation angles and compared by an exact simulated SWAP-test fidelity,
   supplied to the classifier as **one additional feature**. It is switchable
   (`use_quantum=False`) so its contribution is measurable rather than assumed
   -- and measurement shows it contributes nothing.
5. **Per-domain thresholds** calibrated on a group-disjoint validation split,
   never on test.

The architecture is deliberately conventional: it serves as a control, so the
protocol findings above cannot be attributed to architectural tricks.

### Exact simulation, 380x faster

The reference simulator built a 2^n statevector and permuted every dimension
once per qubit, leaving the GPU idle on kernel-launch overhead.
`src/quantum_fast.py` keeps the mathematics exact -- equivalence and gradients
are pinned by tests -- while removing the permutes and precomputing the
entangling layer as a single diagonal phase vector:

| | reference | fast | speedup |
|---|---|---|---|
| CPU | 161 pairs/s | 21,618 pairs/s | 134x |
| GPU (RTX 5060) | 192 pairs/s | 73,094 pairs/s | **380x** |

A closed-form product over qubits does *not* work here: the shared CNOT chain
correlates the Rz phases, so the overlap is not factorizable. The speedup comes
from removing overhead, not from approximating. Note this is quantum-vs-quantum
-- simulating a circuit is always slower than the classical path it sits beside.

---

## Results

Grouped 5-fold cross-validation, every family tested exactly once. No family
or identity appears in both training and test.

| Dataset | Accuracy (95% CI) | ROC-AUC |
| :--- | ---: | ---: |
| KinFaceW-I | **76.92% ± 2.19** | 0.875 |
| KinFaceW-II | **74.25% ± 1.95** | 0.832 |
| FIW | **72.21% ± 1.55** | 0.810 |
| TSKinFace (shortcut-free) | **79.25% ± 2.32** | 0.881 |
| **Mean** | **75.66%** | **0.8497** |

Fold integrity was verified directly: zero folds with train/test group
overlap, every family tested exactly once, no duplicates (95/95 FIW,
533/533 KinFaceW-I, 1000/1000 KinFaceW-II, 559/559 TSKinFace).

**Why no SOTA comparison table?** Published kinship results use each dataset's
official protocol, which permits the identity leakage and TSKinFace shortcut
documented above. Comparing our grouped-k-fold numbers against them would be
misleading in both directions, so we do not present a ranking. A previous
version of this README carried such a table; several of its rows could not be
traced to a verifiable source and have been removed rather than reproduced.

### Efficiency

| Model | Parameters | FIW ROC-AUC |
| :--- | ---: | ---: |
| Siamese ResNet-18 | 11.2 M | 0.685 |
| FaceNet (Inception-ResNet-v1) | 23.5 M | 0.762 |
| ArcFace (ResNet-50) | 31.0 M | 0.814 |
| Vision Transformer (ViT-Base) | 86.4 M | 0.838 |
| **Ours** | **0.50 M** | **0.810** |

Baseline rows are quoted from their source papers under their own protocols
and are **not** directly comparable to our grouped-k-fold numbers; they are
included for scale, not for ranking. Our model is ~170x smaller than ViT-Base.

### Ablation: does the quantum module help?

| | Accuracy | ROC-AUC |
| :--- | ---: | ---: |
| Quantum ON | 75.66% | 0.8497 |
| Quantum OFF | 76.34% | 0.8496 |
| Difference | -0.68 pts | +0.0001 |
| Paired t-test (20 folds) | p = 0.158 | p = 0.982 |

No measurable contribution. The module is retained as a documented ablation,
not as the mechanism behind performance.

> **Note on earlier versions.** A 12-module evaluation suite and an associated
> manuscript previously reported figures such as 77.8% FIW and 83.3% TSKinFace.
> Those were produced under leaked splits and the TSKinFace shortcut described
> above, and are superseded. They are archived in [`legacy/`](legacy/) with an
> explanation; do not cite them.

---

## Repository Layout

```
Quantum_kinship/
├── src/                          # Core library
│   ├── splits.py                 # Family-disjoint splitting (closes identity leakage)
│   ├── kfold.py                  # Grouped k-fold; every family tested exactly once
│   ├── ts_pairs.py               # Shortcut-free TSKinFace negatives
│   ├── multi_dataset.py          # Unified protocol across all 4 datasets
│   ├── models_hybrid.py          # QuantumAugmentedKinshipClassifier
│   ├── quantum_fast.py           # Exact SWAP-test fidelity, 380x faster
│   ├── calibration.py            # FPR-constrained threshold calibration
│   ├── predictor.py              # Deployment inference, per-domain thresholds
│   ├── quantum_core.py           # Reference simulator (validates quantum_fast)
│   ├── models_improved.py        # FaceFeatureExtractor (FaceNet wrapper)
│   └── data_loaders.py           # Dataset parsers
├── scripts/
│   ├── deploy/build_all_caches.py    # Extract FaceNet embeddings (GPU)
│   ├── training/train_multi.py       # Train on all 4 datasets pooled
│   ├── evaluation/run_kfold.py       # Grouped 5-fold evaluation
│   ├── deploy/package_multi.py       # Package model + per-domain thresholds
│   └── inference/predict.py          # CLI: single pair or CSV batch
├── tests/                        # 61 tests
├── results/honest/               # Measured metrics (kfold.json is definitive)
├── weights/
│   ├── deploy/kinship_model.pt   # Deployment artifact
│   └── caches/                   # Precomputed embeddings
├── legacy/                       # Superseded work -- see legacy/README.md
├── RESULTS_HONEST.md             # Full methodology and results
└── README.md
```

---

## Quick Start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt

# GPU (optional but ~15x faster; RTX 50-series needs the cu128 wheel)
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

### Try it on your own photos

```bash
# interactive -- prompts for photos
python verify_kinship.py

# two people, one photo each
python verify_kinship.py --person-a mum.jpg --person-b me.jpg

# several photos each (more accurate: +0.077 ROC-AUC where sets exist)
python verify_kinship.py --person-a dad1.jpg dad2.jpg --person-b kid1.jpg kid2.jpg

# folders and URLs work too
python verify_kinship.py --person-a ./dad_photos/ --person-b https://site/kid.jpg

# both parents plus a child, scored jointly
python verify_kinship.py --father f.jpg --mother m.jpg --child c.jpg
```

Photographs are face-detected and cropped with MTCNN before scoring. This is
required rather than cosmetic: the models were trained on tight face crops, and
an uncropped photograph embeds at cosine **0.027** against its cropped version
--- effectively an unrelated image. Detection restores it to **0.940**. Pass
`--no-detect` only if your inputs are already tight crops.

Photos without a detectable face are reported and skipped, not silently
dropped.

### Predict

```bash
# single pair, one photo each
python scripts/inference/predict.py \
    --img1 parent.jpg --img2 child.jpg --relation fs --domain fiw

# set-level: several photos per person (+0.077 ROC-AUC where sets exist)
python scripts/inference/predict.py \
    --set-a p1.jpg p2.jpg p3.jpg --set-b c1.jpg c2.jpg

# triadic: father + mother + child scored jointly (+0.032 ROC-AUC)
python scripts/inference/predict.py \
    --father f.jpg --mother m.jpg --child c.jpg

# batch from CSV (img1,img2,relation)
python scripts/inference/predict.py --pairs pairs.csv --out results.csv
```

Relations: `fd` father-daughter, `fs` father-son, `md` mother-daughter,
`ms` mother-son. `--domain` selects the calibrated operating point
(`fiw`, `kinfacew-i`, `kinfacew-ii`, `tskinface`).

Shipped models, each scored on a held-out family-disjoint fold:

| Model | Accuracy | ROC-AUC | Use when |
|---|---:|---:|---|
| pairwise | 75.66% | 0.8497 | one photo per person |
| set-level | 73.40% | 0.8228 | several photos per person |
| triadic | 76.58% | 0.8382 | both parents and a child |

The pairwise row averages four datasets under grouped 5-fold; the other two are
single held-out folds on the corpora that support them, so the rows are not
directly comparable. The like-for-like comparison on identical folds is in
`RESULTS_HONEST.md`.

Passing one photo per person to the set interface reproduces the single-image
path exactly, so the set interface is always safe to use.

### Reproduce every number

```bash
python scripts/deploy/build_all_caches.py     # FaceNet embeddings, all datasets
python scripts/evaluation/run_kfold.py        # grouped 5-fold -> results/honest/kfold.json
python scripts/evaluation/run_kfold.py --no-quantum --tag kfold_noq   # ablation
python scripts/training/train_multi.py --tag multi_cap8000 --cap-per-dataset 8000
python scripts/deploy/package_multi.py        # package with per-domain thresholds
pytest tests/ -q                              # 61 tests
```

---

## Citation

```bibtex
@software{quantum_kinship_2026,
  title  = {Leakage-free evaluation for facial kinship verification},
  author = {Vaibhav, Sasi},
  year   = {2026},
  url    = {https://github.com/svaibhav-7/Qiskit_kinship_verification}
}
```

No peer-reviewed publication is associated with this repository yet; please do
not cite it as one.

Distributed under the **MIT License**. See `LICENSE` for details.
