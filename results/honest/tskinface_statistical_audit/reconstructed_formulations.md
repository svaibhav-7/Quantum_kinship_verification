# Reconstructed formulation reruns on corrected TSKinFace

The corrected TSKinFace experiment now evaluates reconstructed versions of
formulations 1--9 under a common family-disjoint protocol. The historical
implementations for formulations 1--7 are not present in this checkout, and
the manuscript does not specify every operational detail. These are therefore
transparent operationalizations, **not exact replications** of the
manuscript's original experiments.

The full per-pair predictions, family identifiers, fold metrics, run
configuration, and statistical results are in
[`formulation_reruns_tskinface_reconstructed_20261105_ci_only.json`](../formulation_reruns_tskinface_reconstructed_20261105_ci_only.json).
The source experiment artifact is preserved separately at
[`formulation_reruns_tskinface_reconstructed_20261105_clustered.json`](../formulation_reruns_tskinface_reconstructed_20261105_clustered.json).
The CI-only artifact supersedes earlier outputs containing fold-based
p-values and multiplicity adjustments. Those tests are not reported because
the five folds partition one family pool and are not independent population
draws.
The family-bootstrap analysis can be regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\recompute_formulation_bootstrap.py `
  results\honest\formulation_reruns_tskinface_reconstructed_20261105_clustered.json `
  --output results\honest\formulation_reruns_tskinface_reconstructed_20261105_ci_only.json
```

## Protocol and scope

- 1,118 positive pair rows from 559 families; all endpoints had cached
  embeddings.
- Five outer folds grouped by family. TSKinFace negatives were generated
  **after** splitting: half father-versus-mother within-family negatives and
  half cross-family negatives restricted to the same split.
- Model training, selection, and thresholding use only family-disjoint
  training/validation groups. Test predictions and endpoint-family IDs are
  retained in the JSON artifact.
- Formulations 2--9 use the same five outer folds. Formulation 1 uses five
  seeded holdouts and is reported descriptively, not as an independent
  significance test. Formulations 3 and 8 reduce to the same singleton-set
  density test here and are not independent results.
- See the artifact's `reconstruction_assumptions` for per-formulation choices.

## Paired results

`Delta` is the mean of five fold-wise treatment-minus-control ROC-AUC values.
The interval samples family clusters at both endpoints within each fold,
computes each fold's paired AUC difference, and averages the folds without
pooling scores from different fitted models. It conditions on fitted models
and is not a significance test.

| Formulation | Reconstructed treatment | Treatment AUC (SD) | Control AUC (SD) | Delta | Family-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| 2 | SWAP-fidelity hybrid | 0.7246 (0.0143) | 0.7061 (0.0320) | +0.0185 | [+0.0045, +0.0325] |
| 3 | Set-density feature | 0.6672 (0.0164) | 0.6675 (0.0187) | -0.0004 | [-0.0037, +0.0029] |
| 4 | Triadic phase-gain feature | 0.7934 (0.0209) | 0.7934 (0.0207) | -0.00004 | [-0.0017, +0.0015] |
| 5 | Entanglement-entropy feature | 0.7047 (0.0124) | 0.7051 (0.0117) | -0.0004 | [-0.0038, +0.0029] |
| 6 | Projective POVM | 0.6189 (0.0239) | 0.6270 (0.0185) | -0.0080 | [-0.0258, +0.0101] |
| 7 | Purity regularizer | 0.6955 (0.0219) | 0.7079 (0.0303) | -0.0123 | [-0.0176, -0.0065] |
| 9 | Triadic phase sweep | 0.7933 (0.0217) | 0.7934 (0.0207) | -0.0001 | [-0.0055, +0.0046] |

Formulation 1's five holdout means were 0.7263 for the quantum arm and 0.7263
for its control (mean paired difference -0.00003); this small holdout sample is
descriptive only. Formulation 8 is the same singleton-set density comparison
as formulation 3.

## Interpretation

These corrected-protocol reconstructions do **not** establish a general
quantum advantage. The SWAP-fidelity arm (formulation 2) has a positive mean
fold difference, but its family-bootstrap interval conditions on the already
fitted fold models; it does not account for training-set or training-seed
variability and must not be presented as population-level confirmation.
Similarly, formulation 7's conditional interval is negative. Neither interval
is a calibrated hypothesis test.

The supportable conclusion is narrow: under these explicit corrected
operationalizations on TSKinFace, set-density, triadic phase features, entropy,
and POVM readout show no reliable improvement over their controls; the
SWAP-fidelity and purity results remain inconclusive. Broader claims require
exact source implementations, repeated outer partitions and training seeds,
and replication on the other corpora under comparably corrected protocols.
