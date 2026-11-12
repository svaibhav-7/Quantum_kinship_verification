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
- **Protocol correction and evidence scope.** The existing grouped 5-fold
  reference evaluation gives a macro-mean ROC-AUC of **0.7770** across the
  four corpora (0.7982 on corrected TSKinFace), below the earlier historical
  mean of 0.8497. Reconstructed formulations 2--9 have also been run on
  corrected TSKinFace as a **separate sensitivity analysis**; these are not
  exact replications and use a different negative construction from protocol
  v2. Formulation 1 is descriptive. See
  [the reconstructed-formulation report](results/honest/tskinface_statistical_audit/reconstructed_formulations.md).
- **Readout is an experimental question, not a superiority claim.** In the
  completed protocol-v2 results across all four corpora and both backbones,
  expectation vectors exceed scalar fidelity and log-fidelity in 16/16
  comparisons by family-bootstrap intervals; against SWAP, 3/16 intervals
  favor SWAP and the rest include zero. Against width-matched controls,
  expectation exceeds the MLP control in 12/16 comparisons. These are
  unadjusted comparison intervals, not a general quantum-advantage claim.

The corrected grouped 5-fold reference results are **71.33% macro-mean
accuracy / 0.7770 macro-mean ROC-AUC**. This report distinguishes them from
historical experiments.

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
   (`use_quantum=False`) so its contribution is measurable rather than assumed.
   The available 20-fold ablation is historical; it predates the TSKinFace
   pair-construction correction and is not a corrected-protocol conclusion.
5. **Per-domain thresholds** calibrated on a group-disjoint validation split,
   never on test.

The architecture is deliberately conventional: it serves as a control, so the
protocol findings above cannot be attributed to architectural tricks.

### Exact SWAP-fidelity simulation

The reference simulator built a 2^n statevector and permuted every dimension
once per qubit. `src/quantum_fast.py` removes those permutes and precomputes
the entangling layer as a diagonal phase vector. Output and input-gradient
equivalence are tested with explicit numerical tolerances; bitwise identity is
not claimed.

Reproduce the forward-only GPU benchmark (preallocated inputs, batch size 256,
six qubits, depth one, three warmups, thirty timed repeats, CUDA synchronization
around each measurement):

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_quantum_fast.py `
  --device cuda --batch-size 256 --qubits 6 --depth 1 `
  --warmup 3 --repeats 30
```

| Device | Reference | Optimized | Speedup |
|---|---:|---:|---:|
| RTX 5060 Laptop GPU | 1,570 pairs/s | 46,316 pairs/s | 29.5x |

A closed-form product over qubits does *not* work here: the shared CNOT chain
correlates the Rz phases, so the overlap is not factorizable. The speedup comes
from removing overhead, not from approximating. This measured 29.5x replaces
the earlier 380x figure, whose batch size and timing procedure were not
preserved. The output and gradient equivalence metrics are saved in
`results/honest/quantum_fast_benchmark_rtx5060.json`. This is quantum-vs-quantum;
simulating a circuit is still slower than the classical path it sits beside.
See [ENVIRONMENT.md](ENVIRONMENT.md) for hardware, software versions, and
numerical tolerances.
---

## Results

Corrected grouped 5-fold reference evaluation, every family tested exactly
once. Accuracy intervals are the runner's 1.96-SE values; ROC-AUC uncertainty
is the standard deviation across folds.

| Dataset | Accuracy (mean ± 1.96 SE) | ROC-AUC (mean ± fold SD) |
| :--- | ---: | ---: |
| KinFaceW-I | **73.00% ± 10.97** | 0.7988 ± 0.1583 |
| KinFaceW-II | **70.85% ± 8.64** | 0.7613 ± 0.1229 |
| FIW | **67.48% ± 7.21** | 0.7495 ± 0.1135 |
| TSKinFace (corrected pairs) | **73.98% ± 4.03** | 0.7982 ± 0.0705 |
| **Macro mean** | **71.33%** | **0.7770** |

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

### Historical ablation: does the quantum module help?

This 20-fold ablation predates the TSKinFace pair-construction correction;
the p-values below are not corrected-protocol estimates.

| | Accuracy | ROC-AUC |
| :--- | ---: | ---: |
| Quantum ON (historical) | 75.66% | 0.8497 |
| Quantum OFF (historical) | 76.34% | 0.8496 |
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
│   ├── quantum_fast.py           # Optimized SWAP-test fidelity
│   ├── calibration.py            # FPR-constrained threshold calibration
│   ├── predictor.py              # Deployment inference, per-domain thresholds
│   ├── quantum_core.py           # Reference simulator (validates quantum_fast)
│   ├── quantum_vqc.py            # Joint-register VQC; both readouts
│   ├── models_vqc.py             # VQC classifiers and their matched controls
│   ├── identity_sets.py          # Person-level (photo-set) representation
│   ├── triads.py                 # Father-mother-child triadic scoring
│   ├── models_improved.py        # FaceFeatureExtractor (FaceNet wrapper)
│   └── data_loaders.py           # Dataset parsers
├── scripts/
│   ├── deploy/build_all_caches.py    # Extract FaceNet embeddings (GPU)
│   ├── training/train_multi.py       # Train on all 4 datasets pooled
│   ├── evaluation/run_kfold.py       # Historical evaluation; not protocol v2
│   ├── evaluation/run_amplitude_vqc.py  # Historical readout ablation
│   ├── evaluation/make_figures*.py   # All paper figures, from artefacts
│   ├── deploy/package_multi.py       # Package model + per-domain thresholds
│   └── inference/predict.py          # CLI: single pair or CSV batch
├── paper/                        # Two manuscripts; see paper/README.md
├── tests/                        # 290 tests
├── results/honest/               # Measured metrics and correction audit
├── weights/
│   ├── deploy/kinship_model.pt   # Deployment artifact
│   └── caches/                   # Precomputed embeddings
├── legacy/                       # Superseded work -- see legacy/README.md
├── RESULTS_HONEST.md             # Full methodology and results
├── ENVIRONMENT.md                # Exact versions, hashes, seeds, commands
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

# several photos each (set-level model; performance varies by dataset)
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

### Visualize random dataset examples

```bash
python verify_kinship.py --dataset-samples --dataset FIW --samples 12 --fold 1 --seed 42
```

This trains the protocol-v2 fair reference on family-disjoint fit and validation
splits, constructs relation-matched negatives after splitting, and scores only
the selected held-out fold. The PNG is saved under `results/demo/` by default.
Choose `KinFaceW-I`, `KinFaceW-II`, or `TSKinFace` with `--dataset` to sample
another available dataset. The displayed examples are illustrative, not a
benchmark estimate; this mode trains a corpus-specific demo model.

### Predict

```bash
# single pair, one photo each
python scripts/inference/predict.py \
    --img1 parent.jpg --img2 child.jpg --relation fs --domain fiw

# set-level: several photos per person (performance varies by dataset)
python scripts/inference/predict.py \
    --set-a p1.jpg p2.jpg p3.jpg --set-b c1.jpg c2.jpg

# triadic: father + mother + child scored jointly (experimental)
python scripts/inference/predict.py \
    --father f.jpg --mother m.jpg --child c.jpg

# batch from CSV (img1,img2,relation)
python scripts/inference/predict.py --pairs pairs.csv --out results.csv
```

Relations: `fd` father-daughter, `fs` father-son, `md` mother-daughter,
`ms` mother-son. `--domain` selects the calibrated operating point
(`fiw`, `kinfacew-i`, `kinfacew-ii`, `tskinface`).

Historical shipped-artifact scores (not corrected-protocol reference results):

| Model | Accuracy | ROC-AUC | Use when |
|---|---:|---:|---|
| pairwise | 75.66% | 0.8497 | one photo per person |
| set-level | 73.40% | 0.8228 | several photos per person |
| triadic | 76.58% | 0.8382 | both parents and a child |

The pairwise row averages four datasets under the earlier grouped 5-fold
protocol, including the uncorrected TSKinFace pairs; the other two are single
held-out folds on the corpora that support them, so the rows are not directly
comparable. The corrected reference baseline is reported above; the
like-for-like historical comparison on identical folds is in `RESULTS_HONEST.md`.

Passing one photo per person to the set interface reproduces the single-image
path exactly, so the set interface is always safe to use.

### Reproduce current protocol-v2 results

```bash
.\.venv\Scripts\python.exe scripts\deploy\build_rfiw_cache.py
.\.venv\Scripts\python.exe scripts\evaluation\run_protocol_v2.py all --output-dir results\protocol_v2_clean
.\.venv\Scripts\python.exe scripts\evaluation\summarize_protocol_v2.py --input-dir results\protocol_v2_clean
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The standalone `run_kfold.py`, `run_vqc.py`, `run_amplitude_vqc.py`,
`run_triadic.py`, `run_setlevel.py`, `run_backbone_comparison.py`, and
`run_scaling.py` scripts are historical evaluations, not protocol-v2
reproduction commands. Existing
`results/protocol_v2/` outputs were generated before the endpoint-family
bootstrap and matched no-shared baseline fixes; retain them as exploratory.
Run the current protocol from a clean source commit into a new output
directory, and do not describe the rerun as complete until its artifacts and
summary have been checked.

---

---

---

## Project History and Methodology

This repository began as a quantum-inspired kinship system reporting 77.8% on
FIW. Re-examination found that number was measured with 100% identity leakage,
and that nine quantum-inspired formulations each failed against a
capacity-matched classical control. A tenth formulation, differing from the
nine only in how the quantum state is read out, then showed that those nine
failures were confounded with the readout itself.

**[docs/project_history.pdf](docs/project_history.pdf)** records the full route:
the diagnostics that prompted the change, why each quantum formulation failed
and what that implies structurally, the methodology that replaced them, and one
claim this project made and later had to retract.

In brief:

| | Then | Now |
|---|---|---|
| FIW | 76.0% (100% identity leakage) | 72.21% ± 1.55 (family-disjoint) |
| TSKinFace | 83.3% (photo-session shortcut) | 79.25% ± 2.32 (shortcut closed) |
| Seed stability | 22.5-point spread | 12.1 points |
| Quantum claim | "removing it costs 7.5%" | 9 scalar-readout formulations fail; the readout is the bottleneck |

The numbers went down. That drop is the leakage and the shortcut leaving.

### Manuscripts

Two papers are written from this work; both compile from `paper/` (verified
with `pdflatex -halt-on-error`, zero undefined references) and every numeric
claim traces to a JSON artefact in `results/honest/`. `tests/test_manuscripts.py`
guards the source against defects that reach a PDF silently.

| Path | Leads with |
|---|---|
| [paper/elsevier/quantum_negative.pdf](paper/elsevier/quantum_negative.pdf) | the readout bottleneck: ten formulations, and why nine of them failed |
| [paper/ieee/main.pdf](paper/ieee/main.pdf) | person-level verification under leakage-free evaluation |
| [paper/elsevier_person/main.pdf](paper/elsevier_person/main.pdf) | the same paper as `ieee/`, typeset for Elsevier (generated) |

An earlier headline in this project -- that the quantum arm beat its
capacity-matched control by +0.132 ROC-AUC -- did not survive scrutiny: the
control had been matched on parameter count, which collapsed it to a
one-dimensional readout. Against controls matched on readout width the
comparison is inconsistent across corpora and backbones, and it is no longer
claimed. The readout result, where both arms are quantum, is unaffected and
replicates 8/8.

---

## Citation


No peer-reviewed publication is associated with this repository yet; please do
not cite it as one.

Distributed under the **MIT License**. See `LICENSE` for details.
