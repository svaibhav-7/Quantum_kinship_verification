# Honest Evaluation Results

All numbers below are measured on **family-disjoint** FIW splits: no family,
and therefore no identity, appears in both training and test. They are **not**
comparable to earlier figures in this repo, which were measured with complete
identity leakage.

## 1. The leak that invalidated prior numbers

FIW has 95 families and 472 identities spread over 57,120 pairs; one family
(`F0282`) alone is 19% of all pairs, and identities recur 242x on average.
The previous code split on *pair index*, which produced:

```
test pairs sharing an identity with train : 11424/11424 = 100.0%
test pairs whose family was seen in train : 11424/11424 = 100.0%
```

Every metric produced that way had the answer key in the training set.

`src/splits.py` splits on family instead. Negatives must be generated *after*
the split: each non-kin pair bridges two families, and on real FIW that
collapses all 95 families into a single connected component, making a disjoint
split impossible. Verified on the full dataset: 0 shared families,
0 shared identities, 20.0% test, 1:1 balanced.

## 2. The quantum fidelity was the bottleneck

On identical embeddings and pairs:

| Method | Accuracy | ROC-AUC |
|---|---|---|
| Raw cosine similarity (no training) | 68.0% | 0.734 |
| Logistic regression on `[abs_diff, product, cosine]` | 75.1% | 0.821 |
| Original quantum meta-ensemble | 59.0% | 0.591 |

A trivial baseline beat the full quantum pipeline by 16 points. Routing every
decision through one `product-of-cos^2` scalar discarded the signal; gradient
norms confirmed it (projection MLP ~2e1, quantum params ~4e-2).

`src/models_hybrid.py` keeps the fidelity as **one input feature** beside
symmetric pair features, so the quantum branch retains a real gradient path
and stays switchable for ablation.

## 3. Quantum ablation: 5 seeds, paired

| Seed | Quantum ON | Quantum OFF | Diff |
|---|---|---|---|
| 1 | 73.63% | 72.57% | +1.07 |
| 2 | 79.76% | 79.06% | +0.70 |
| 3 | 81.04% | 77.87% | +3.17 |
| 4 | 73.70% | 78.28% | -4.58 |
| 5 | 72.86% | 72.75% | +0.11 |
| **mean** | **76.20% +/- 3.88** | **76.11% +/- 3.18** | **+0.09** |

Paired t-test: **t = 0.074, p = 0.945**.

**The quantum module contributes no measurable accuracy.** This is a genuine
negative result, not a bug: the branch is exercised (tests assert that
disabling it changes predictions and that gradients reach `ent_params`).
It should be reported as an ablation, not as the mechanism driving performance.

## 4. Deployment model

| Metric | Value |
|---|---|
| Held-out accuracy | **82.49%** |
| Held-out ROC-AUC | **0.8990** |
| Recall (TPR) | 78.81% |
| False-positive rate | 13.84% |
| Precision | 85.07% |
| Test pairs | 11,420 |
| Threshold (calibrated on validation) | 0.2119 |

Youden's J was the wrong calibration objective here: it maximises TPR-FPR and
produced a **24.3% false-positive rate** -- one in four unrelated pairs called
kin. `src/calibration.py` maximises accuracy subject to an explicit FPR
ceiling, which improved both axes at once (81.94% -> 82.49% accuracy,
24.3% -> 13.8% FPR). The test set is touched once, after model selection.

### Verified end-to-end on real photographs

Freshly extracted embeddings match the training-time cache to
**cosine 1.000000**, so there is no train/deploy preprocessing skew and the
held-out number transfers to raw JPEG input.

### The same-photo shortcut does not explain the result

8.2% of FIW kin pairs come from the *same source photograph* (vs 0% of
non-kin), which would let a model key on shared lighting and background.
Removing those pairs from the test set changes nothing:

| Test set | Accuracy | ROC-AUC |
|---|---|---|
| With same-photo pairs | 82.53% | 0.8990 |
| **Without same-photo pairs** | **82.61%** | 0.8986 |

The model is not exploiting the shortcut. (TSKinFace, by contrast, is 100%
same-family positives -- there the shortcut is the whole signal.)

### Reproducibility note

Regularization matters more than data volume here. Weakening dropout 0.5->0.4
and weight decay 1e-2->5e-3 cost ~8 points (76% -> 68% mean over 3 seeds).
The reported settings are dropout 0.5, weight decay 1e-2, early stopping on a
family-disjoint validation split.

## 5. Performance

The reference simulator built a 2^n statevector and permuted all n dimensions
per qubit, launching thousands of tiny kernels -- so the GPU sat idle.
`src/quantum_fast.py` keeps the exact same maths (equivalence and gradients
pinned by tests) with no permutes and a single precomputed phase vector:

| | reference | fast | speedup |
|---|---|---|---|
| CPU | 161 pairs/s | 21,618 pairs/s | 134x |
| GPU (RTX 5060) | 192 pairs/s | 73,094 pairs/s | **380x** |

Note: a closed-form product over qubits does *not* work here -- the shared
CNOT chain correlates the Rz phases, so the overlap is not factorizable.
(Verified: with Rz disabled a product form matches exactly; with distinct Rz
it diverges.) The speedup comes from removing overhead, not from approximating.

## 6. Multi-dataset results (final)

The FIW-only model did not generalise: 82.5% on FIW but 61-65% on datasets it
had never trained on. It had learned FIW, not kinship. Training on all four
datasets pooled (capped at 8,000 pairs each, so FIW's 94% share cannot swamp
the rest) fixed this:

| Dataset | n | Accuracy | ROC-AUC | FPR |
|---|---:|---:|---:|---:|
| FIW | 11,420 | 75.10% | 0.8735 | 7.4% |
| KinFaceW-I | 212 | 83.96% | 0.8982 | 17.0% |
| KinFaceW-II | 400 | 71.00% | 0.8434 | 11.5% |
| TSKinFace (shortcut-free) | 555 | 85.41% | 0.9177 | 14.1% |
| **Mean** | | **78.87%** | **0.8832** | 12.5% |

Cross-dataset transfer, FIW-only model vs multi-dataset model:

| Dataset | FIW-only | Multi-dataset | Gain |
|---|---:|---:|---:|
| KinFaceW-I | 62.38% | 83.96% | **+21.6** |
| KinFaceW-II | 64.80% | 71.00% | **+6.2** |
| TSKinFace (shortcut-free) | 60.73% | 85.41% | **+24.7** |

Thresholds are calibrated **per domain** on a group-disjoint validation slice;
a single global threshold cost between 0.7 and 5.9 points depending on dataset.

### TSKinFace: shortcut closed

Stock TSKinFace makes every positive same-family and every negative
cross-family, so "same photo session" separates the classes perfectly.
`src/ts_pairs.py` draws half the negatives *inside* a family (father vs
mother -- co-photographed, not kin):

| Pair source | same-family positives | same-family negatives | cosine ROC-AUC |
|---|---:|---:|---:|
| Stock loader | 100% | 0% | 0.7278 |
| Shortcut-free | 100% | 50% | 0.6880 |

The ~9-point AUC drop is the shortcut being removed. All TSKinFace numbers
reported here use the harder, shortcut-free pairs.

### Two mistakes worth recording

1. **Inner validation leaked.** A random slice of pooled training pairs shares
   families with the fit set: validation AUC read 0.995 while true test AUC
   was 0.844. Validation is now group-disjoint too.
2. **Group-disjoint splitting discarded data.** Fixed negatives that join two
   families straddle any boundary, so KinFaceW-I lost 173/1066 pairs and left
   only 119 for test. `split_rebuild_negatives` rebuilds negatives per side
   instead: KinFaceW-I test recovered 119 -> 212, KinFaceW-II 242 -> 400.

## 7. README claims corrected


## 8. Grouped k-fold (definitive numbers)

Single group-disjoint splits were unstable: accuracy ranged 61.7%-84.1%
across seeds because FIW's test side was effectively two families, one of
which (F0282) held 95% of its pairs. Grouped 5-fold puts **every family in
the test set exactly once** and caps any family at 1,500 pairs.

| Dataset | Folds | Accuracy (95% CI) | ROC-AUC |
|---|---:|---:|---:|
| FIW | 5 | 72.21% ± 1.55 | 0.8101 ± 0.0175 |
| KinFaceW-I | 5 | 76.92% ± 2.19 | 0.8752 ± 0.0146 |
| KinFaceW-II | 5 | 74.25% ± 1.95 | 0.8320 ± 0.0190 |
| TSKinFace | 5 | 79.25% ± 2.32 | 0.8814 ± 0.0192 |

**Mean across datasets: 75.66% accuracy, 0.8497 ROC-AUC.**

Stability, single-split vs k-fold (accuracy range):

| Dataset | Single-split | k-fold |
|---|---|---|
| FIW | 61.7-75.9% (14.2) | 69.8-74.4% (4.6) |
| KinFaceW-I | 70.3-78.8% (8.5) | 72.9-79.4% (6.5) |
| KinFaceW-II | 69.0-77.8% (8.8) | 71.0-77.0% (6.0) |
| TSKinFace | 75.1-84.1% (9.0) | 75.7-81.9% (6.2) |

Overall spread fell from 22.5 points to 12.1. Fold integrity was verified
directly: 0 folds with train/test group overlap, every family tested exactly
once, no duplicates (95/95 FIW, 533/533 KFW-I, 1000/1000 KFW-II, 559/559 TS).

### Quantum ablation under k-fold

20 paired observations (4 datasets x 5 folds):

| | Accuracy | ROC-AUC |
|---|---:|---:|
| Quantum ON | 75.66% | 0.8497 |
| Quantum OFF | 76.34% | 0.8496 |
| Difference | -0.68 pts | +0.0001 |
| Paired t-test | p = 0.158 | p = 0.982 |

The quantum module contributes **nothing measurable**, confirming the earlier
5-seed result (p = 0.945) on a far more stable protocol. It is retained as a
documented ablation, not as the mechanism behind performance.

### Audit findings

- **No hardcoded metrics.** Contrast `build_meta_ensemble.py:99`, which
  returned invented accuracies `(57.63, 65.62, 88.12)` to derive weights.
- **No leakage.** Shuffled-label control: real 0.864 AUC vs shuffled 0.508.
- **No pathological overfitting.** fit 0.966 -> val 0.926 -> test 0.869.
- **Fixed:** `fd_003` names different families in KinFaceW-I and -II; the
  unscoped group key merged them. Now dataset-scoped.


## 10. Set-level and triadic representations

Two ways the task is conventionally *posed* discard available evidence.

### Set-level identity representation

The protocol scores one photograph against one photograph, but FIW supplies a
median of five per identity.

| Dataset | Set size | Single | Set-level | Delta |
|---|---:|---:|---:|---:|
| KinFaceW-I | 1.00 | 0.7210 | 0.7210 | 0.0000 |
| KinFaceW-II | 1.00 | 0.7672 | 0.7672 | 0.0000 |
| TSKinFace | 1.00 | 0.7306 | 0.7306 | 0.0000 |
| **FIW** | **7.30** | 0.7296 | **0.8066** | **+0.0771** |

The gain appears only where photo sets exist. The three single-photograph
corpora are bit-identical to the baseline, which is the degradation guarantee
holding exactly. `+0.0771` closely matches the untrained probe estimate of
`+0.071`, so the gain is intrinsic to the representation rather than extra
model capacity.

**This is a large gain on one benchmark and exactly zero on three.** Any
average over the four obscures that, so we report per dataset.

### Triadic scoring (TSKinFace)

| Arm | Accuracy (95% CI) | ROC-AUC |
|---|---:|---:|
| Best single parent | 67.62 ± 2.69 | 0.7569 |
| **Triadic** | **69.05 ± 2.65** | **0.7893** |
| Triadic + phase sweep | 69.50 ± 2.41 | 0.7887 |

Triadic scoring gains **+0.0324 ROC-AUC** over the best single parent
(paired t-test over folds, t = 14.93, **p = 0.0001**, positive in all 5 folds).
The strongest single contributor is `cos(F,M)` -- parental similarity -- which
no pairwise model can express.

### A bug caught before it became a result

The first set-level run reported +9.8 AUC and KinFaceW rising 0.725 to 0.898.
Set sizes gave it away: 55.5 images per identity on KinFaceW-I, which stores
exactly one. `identity_of_path` was FIW-specific and returned the *directory*
on other datasets, collapsing all 533 KinFaceW-I people into 4 buckets. The
key is now dataset-aware, pinned by four tests.

### Formulations 8 and 9

| Formulation | Control | Result |
|---|---|---|
| Density fidelity on photo sets | set features | -0.0006, -0.0005, +0.0001, -0.0006 |
| Interference phase sweep on triads | convex mixture | -0.0006, p = 0.147 |

Nine formulations, none beating its classical control.

### Shipped deployment models

Both new components are packaged and wired into the CLI. Each was fitted on a
family-disjoint training split, calibrated on a family-disjoint validation
slice, and scored once on a held-out fold.

| Artifact | Accuracy | ROC-AUC | Threshold |
|---|---:|---:|---:|
| `weights/deploy/kinship_model.pt` (pairwise) | 75.66% | 0.8497 | per-domain |
| `weights/deploy/set_model.pkl` | 73.40% | 0.8228 | 0.4987 |
| `weights/deploy/triad_model.pkl` | 76.58% | 0.8382 | 0.5159 |

The set-level accuracy here (73.40%) sits below the pairwise 75.66%, and the
two are **not** comparable: the pairwise figure averages four datasets under
grouped 5-fold, while set-level is a single held-out FIW fold — FIW being the
only corpus with photo sets, and the hardest of the four. The like-for-like
comparison on identical folds is Section 10: FIW 0.7296 → 0.8066.

