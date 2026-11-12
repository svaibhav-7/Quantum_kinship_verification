# Evaluation Results: Historical and Corrected Evidence

This report preserves results from several experiment generations; its
historical sections are not all corrected-protocol estimates. After correcting
TSKinFace negative-pair construction, the grouped-five-fold reference macro
mean is ROC-AUC **0.77696** (not the earlier 0.8497). Of formulations 1--9,
reconstructed formulations 2--9 have corrected-TSKinFace sensitivity runs;
they are not exact replications and use a different negative construction
from protocol v2. Formulation 1 is descriptive. The readout analysis has
corrected TSKinFace folds only, while its other-corpus cells remain historical.
Fold-based p-values are not treated as calibrated evidence. Consult
[`results/honest/tskinface_statistical_audit/README.md`](results/honest/tskinface_statistical_audit/README.md)
for provenance, corrected values, and inferential limitations; see also the
[reconstructed-formulation sensitivity report](results/honest/tskinface_statistical_audit/reconstructed_formulations.md).
The sections below are a historical record except where
explicitly marked corrected.

## Final corrected protocol-v2 experiments (2026-11-12)

This section summarizes the final grouped-family protocol-v2 run. The complete
tables and per-comparison intervals are in
[`SUMMARY.md`](results/protocol_v2_corrected_20261110/SUMMARY.md). It contains
all 360 reference records per backbone, all 480 readout records per backbone,
and zero incomplete comparisons. Fold summaries use descriptive Student-t
half-widths; paired AUC differences use two-endpoint family-bootstrap
confidence intervals. No bootstrap p-values are reported.

### Main comparisons

| Comparison | Corrected result |
|---|---|
| FIW set vs single, FaceNet | Δ AUC +0.0734; 95% family-bootstrap CI [+0.0559, +0.0911] |
| FIW set vs single, ArcFace | Δ AUC +0.0957; 95% CI [+0.0792, +0.1128] |
| FIW set vs single with shared-source pairs excluded, FaceNet | Δ AUC +0.0781; 95% CI [-0.0081, +0.1634] |
| FIW set vs single with shared-source pairs excluded, ArcFace | Δ AUC +0.0703; 95% CI [-0.0136, +0.1472] |
| TSKinFace triadic vs best parent, FaceNet | Δ AUC +0.0341; 95% CI [+0.0137, +0.0548] |
| TSKinFace triadic vs best parent, ArcFace | Δ AUC +0.0287; 95% CI [+0.0159, +0.0420] |

The no-shared-source set-level intervals include zero for both backbones, so
the point-estimate gains do not establish a shared-photo-independent gain.
The hybrid quantum-on/off comparison is also not uniform: all eight FaceNet
intervals include zero; for ArcFace, the FIW intervals include zero while the
KinFaceW-I, KinFaceW-II, and TSKinFace intervals favor quantum-on. Treat these
as separate comparisons, not as multiplicity-adjusted tests.

For the readout ablation, expectation readout beats scalar fidelity and
log-fidelity in all 16 backbone/corpus/view comparisons by the reported
family-bootstrap intervals. Against SWAP, only 3 of 16 intervals exclude zero.
Against width-matched controls, expectation beats the linear control in 16/16
comparisons, the MLP control in 12/16, and the random control in 16/16. The
full intervals and deltas are in the generated summary.

### Negative-sampling sensitivity and limitations

The two additional reference draws use offsets 1000 and 2000, with fixed model
seeds and folds, and share one protocol source fingerprint. Their descriptive
comparison covers 80 common AUC-difference rows. Across the 64 reference
comparisons, the mean SD across these two draws is 0.0051 AUC (maximum 0.0145);
across the 16 quantum-on/off formulation comparisons, it is 0.0054 (maximum
0.0207). These are two-draw sensitivity summaries, not inferential intervals.
The original offset-0 run is reported separately because it has a different
full source fingerprint after the bootstrap implementation was optimized.
See
[`negative_sampling_sensitivity.md`](results/protocol_v2_sensitivity_20261112_offset1000_parallel/negative_sampling_sensitivity.md)
and the per-draw
[`offset-1000 summary`](results/protocol_v2_sensitivity_20261112_offset1000_parallel/SUMMARY.md)
and
[`offset-2000 summary`](results/protocol_v2_sensitivity_20261112_offset2000_parallel/SUMMARY.md).

Thirty of 75 ArcFace model fits and 10 of 75 FaceNet fits were flagged for
validation AUC below their untrained cosine baseline. These are completed
fits that underperformed the validation baseline, not crashed runs; the
comparison is documented in the full summary.

The official RFIW test result is based on the 29,357 of 39,743 pairs covered
by the local face files and embedding cache (73.9%). Coverage is 79.6% for kin
pairs and 68.1% for non-kin pairs; the kin-minus-non-kin difference is +11.52
percentage points (95% two-endpoint family-cluster CI [+8.67, +14.84]).
Availability also varies by relation (for example, 37.9% for grandmother-
granddaughter pairs). Thus RFIW findings apply to the covered subset; missing
pairs are not a neutral random sample. The full audit is in
[`rfiw_test_coverage.md`](results/honest/rfiw_test_coverage.md).

The numbered sections that follow retain historical results and corrections;
do not treat them as protocol-v2 estimates unless explicitly stated.

The original FIW leakage analysis uses **family-disjoint** splits: no family,
and therefore no identity, appears in both training and test.

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

## 5. SWAP-fidelity simulation throughput

The reference simulator built a 2^n statevector and permuted all n dimensions
per qubit. The updated benchmark measures forward inference only, with inputs
preallocated on-device, three warm-up batches, thirty timed calls, median
wall-clock latency, and CUDA synchronization around each call. Outputs and
input gradients are checked with tolerances (`rtol=1e-4, atol=1e-5` and
`rtol=1e-3, atol=1e-4`, respectively); bitwise identity is not claimed.

| Device | Batch | Qubits / depth | Reference | Fast | Speedup |
|---|---:|---:|---:|---:|---:|
| RTX 5060 Laptop GPU | 256 | 6 / 1 | 1,570 pairs/s | 46,316 pairs/s | 29.5x |

Note: a closed-form product over qubits does *not* work here -- the shared
CNOT chain correlates the Rz phases, so the overlap is not factorizable.
(Verified: with Rz disabled a product form matches exactly; with distinct Rz
it diverges.) The speedup comes from removing overhead, not from approximating.
Reproduce with
`.\.venv\Scripts\python.exe scripts\benchmark_quantum_fast.py --device cuda --batch-size 256 --qubits 6 --depth 1 --warmup 3 --repeats 30`.
Hardware/software versions and full output are recorded in
[`ENVIRONMENT.md`](ENVIRONMENT.md) and
`results/honest/quantum_fast_benchmark_rtx5060.json`. This replaces the earlier
380x figure, whose benchmark settings were not preserved.

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


## 8. Corrected grouped k-fold reference numbers

Single group-disjoint splits were unstable: accuracy ranged 61.7%-84.1%
across seeds because FIW's test side was effectively two families, one of
which (F0282) held 95% of its pairs. Grouped 5-fold puts **every family in
the test set exactly once** and caps any family at 1,500 pairs.

| Dataset | Folds | Accuracy (mean ± 1.96 SE) | ROC-AUC (mean ± fold SD) |
|---|---:|---:|---:|
| FIW | 5 | 67.48% ± 7.21 | 0.7495 ± 0.1135 |
| KinFaceW-I | 5 | 73.00% ± 10.97 | 0.7988 ± 0.1583 |
| KinFaceW-II | 5 | 70.85% ± 8.64 | 0.7613 ± 0.1229 |
| TSKinFace (corrected pairs) | 5 | 73.98% ± 4.03 | 0.7982 ± 0.0705 |

**Macro mean across datasets: 71.33% accuracy, 0.77696 ROC-AUC.**

Fold integrity was verified directly: 0 folds with train/test group overlap,
every family tested exactly once, no duplicates (95/95 FIW, 533/533 KFW-I,
1000/1000 KFW-II, 559/559 TSKinFace). Fold intervals are descriptive:
the folds partition the same identity pool and are not independent population
samples.

### Historical quantum ablation under k-fold

The following 20-fold ablation predates the TSKinFace correction and is not a
corrected-protocol comparison:

| | Accuracy | ROC-AUC |
|---|---:|---:|
| Quantum ON | 75.66% | 0.8497 |
| Quantum OFF | 76.34% | 0.8496 |
| Difference | -0.68 pts | +0.0001 |
| Paired t-test | p = 0.158 | p = 0.982 |

This historical ablation does not show a measurable contribution (p = 0.982),
but its TSKinFace result predates the pair-construction correction. It is not
confirmation under the corrected protocol.

### Audit findings

- **No hardcoded metrics.** Contrast `build_meta_ensemble.py:99`, which
  returned invented accuracies `(57.63, 65.62, 88.12)` to derive weights.
- **No leakage.** Shuffled-label control: real 0.864 AUC vs shuffled 0.508.
- **No pathological overfitting.** fit 0.966 -> val 0.926 -> test 0.869.
- **Fixed:** `fd_003` names different families in KinFaceW-I and -II; the
  unscoped group key merged them. Now dataset-scoped.


## 10. Historical set-level and triadic representations

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

The TSKinFace pairwise row and the results in this section predate the
pair-construction correction; they are retained as historical measurements.

The gain appears only where photo sets exist. The three single-photograph
corpora are bit-identical to the baseline, which is the degradation guarantee
holding exactly. `+0.0771` closely matches the untrained probe estimate of
`+0.071`, so the gain is intrinsic to the representation rather than extra
model capacity.

**This is a large gain on one benchmark and exactly zero on three.** Any
average over the four obscures that, so we report per dataset.

### Historical triadic scoring (TSKinFace)

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

### Historical formulations 8 and 9

| Formulation | Control | Result |
|---|---|---|
| Density fidelity on photo sets | set features | -0.0006, -0.0005, +0.0001, -0.0006 |
| Interference phase sweep on triads | convex mixture | -0.0006, p = 0.147 |

These are historical values. Separately, reconstructed formulations 2--9
were evaluated on corrected TSKinFace pairs as a sensitivity analysis. They
use a different negative construction from protocol v2 and are not exact
replications. Their family-bootstrap intervals condition on the fitted
models; the report deliberately presents intervals only, not fold-based
p-values or multiplicity adjustments. See
[`reconstructed_formulations.md`](results/honest/tskinface_statistical_audit/reconstructed_formulations.md).

### Historical shipped deployment models

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
