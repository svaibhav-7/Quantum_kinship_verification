# Evaluation protocol v2 and analysis plan

Written before the v2 reruns. This is an internal analysis plan, not an external
pre-registration. Some matched-negative test scores had already been seen during
the audit that motivated this protocol, so the local-fold results are not blind.
The official RFIW test set had not been scored by any model when this was written.

## Why v2 exists

Family-disjoint splits stopped identity leakage but left label leakage:

| Problem found | Evidence |
|---|---|
| Runner negatives ignored the relation label | KinFaceW: ~50% wrong generation, ~51% contradict the label's gender |
| TSKinFace father-mother negatives were always `fs`/`fd` | Relation label alone: 0.62 ROC-AUC |
| Validation slices were mostly positive | ~84% positive, skewing selection and thresholds |
| Relation vector added to unit-norm embeddings | Validation AUC 0.507 with it, 0.760 without it |
| One dead training run set the headline | Fold 3 near chance; three other seeds reached ~0.80-0.85 |
| Cross-family negatives reintroduce photo context | Colour alone: TSKinFace 0.729, KinFaceW-II 0.732, KinFaceW-I 0.624, FIW 0.555 |

## Rules

1. **Splits.** Five family-grouped folds on positives. Fit and validation are split
   on positives inside the training side. FIW is capped at 1,500 pairs per family.
   Readout runs keep their historical cap of 200.
2. **Negatives.** `src/fair_protocol.py`: same parent, a same-relation child from
   another family on the same side. These remove relation, generation and gender
   cues, but not photo context, so every result is also scored on **colour-matched**
   negatives (`build_side(..., distance=ColourDistance())`). Colour is a proxy for a
   photo session, so this limits that cue without proving it is gone. A colour-only
   arm reports how much context each view still carries. A corpus is treated as
   evidence about kinship only where colour-only scores near 0.5 on that view. The
   older negatives are scored only to measure the inflation.
3. **Duplicates.** Families whose full image set repeats another family are removed
   (TSKinFace `FMD-216` duplicates `FMD-205`).
4. **Seeds.** Every trained arm runs on seeds 0, 1 and 2 for every fold.
   `--negative-seed-offset` can produce a separate negative draw while keeping
   family folds and model initialization seeds fixed. Save each offset to a
   separate output directory; do not resume one directory with another offset.
5. **Failed runs.** A run is flagged when its best validation AUC is more than 0.02
   below untrained cosine on the same validation set. Primary results include all
   runs; a sensitivity result excludes flagged ones. The rule reads validation only.
6. **Thresholds.** Calibrated on balanced validation data, never on test.
7. **Model choice.** Early stopping and all model selection use validation AUC.
   `FairKinshipModel` was chosen on validation data from the training side of fold 1
   only. Those families are test data in other folds, so local-fold comparisons
   involving it are mildly optimistic. RFIW test is unaffected.
8. **Uncertainty.** The current summary reports confidence intervals only.
   The bootstrap resamples both endpoint families of each dyad and computes
   weighted paired AUC differences on within-fold ranks. It conditions on the
   fitted models and is not a hypothesis test; no bootstrap p-values or
   Holm/BH adjustments are reported. Fold intervals use Student's t and are
   descriptive. Variation across negative-sampling draws is summarized
   separately and is not folded into the family-bootstrap intervals.
9. **Provenance.** Every record stores the commit, dirty-tree flag, source-tree
   SHA-256, negative-seed offset and model seed. Fold-level records and scores
   are always saved.

## Primary questions

| # | Question | Comparison | Test family |
|---|---|---|---|
| Q1 | What is kinship performance on fair negatives? | Reference arms vs cosine, per corpus | reference |
| Q2 | How much did the old negatives inflate scores? | Old-model AUC on old vs matched negatives | descriptive |
| Q3 | Is a vector readout better than a scalar? | Expectation vs fidelity, log-fidelity and SWAP scalars | readout |
| Q4 | Does the circuit beat classical controls of equal width? | Expectation vs each width control | control |
| Q5 | Do formulations 1-9 beat their controls? | Each formulation vs its control | formulations |
| Q6 | Does the photo-set gain survive removing shared photos? | Set vs single on matched retained photo-pair rows; shared-source query pairs are excluded from both no-shared arms | set-level |
| Q7 | Does triadic scoring help? | Triad vs best parent; phase vs no phase | triadic |
| Q8 | How does this compare with published FIW work? | Official RFIW Track-I test set | descriptive |
| Q9 | What does the circuit cost? | Timed inference and training with recorded settings | descriptive |
| Q10 | How much of each score is photo context? | Colour-only arm; cosine on matched vs colour-matched | descriptive |

## Reconstruction caveat

The original code for formulations 1-7 is not in this checkout. The v2 versions reuse
the documented reconstructions in `scripts/evaluation/rerun_formulations_tskinface.py`,
so they re-test the ideas rather than exactly replicate the historical runs.

## Reproducing

```powershell
.\.venv\Scripts\python.exe scripts\deploy\build_rfiw_cache.py
.\.venv\Scripts\python.exe scripts\evaluation\run_protocol_v2.py all --output-dir results\protocol_v2_clean
.\.venv\Scripts\python.exe scripts\evaluation\summarize_protocol_v2.py --input-dir results\protocol_v2_clean
# Repeat the primary reference comparison with independent negative draws.
.\.venv\Scripts\python.exe scripts\evaluation\run_protocol_v2.py reference --negative-seed-offset 1000 --output-dir results\protocol_v2_neg1000
.\.venv\Scripts\python.exe scripts\evaluation\summarize_protocol_v2.py --input-dir results\protocol_v2_neg1000
.\.venv\Scripts\python.exe scripts\evaluation\run_protocol_v2.py reference --negative-seed-offset 2000 --output-dir results\protocol_v2_neg2000
.\.venv\Scripts\python.exe scripts\evaluation\summarize_protocol_v2.py --input-dir results\protocol_v2_neg2000
.\.venv\Scripts\python.exe scripts\evaluation\compare_negative_sampling.py results\protocol_v2_clean results\protocol_v2_neg1000 results\protocol_v2_neg2000
# Stage updated deployment artifacts without replacing the current model.
.\.venv\Scripts\python.exe scripts\deploy\package_v2.py --output-dir weights\deploy\v2_candidate
```

Runs are resumable. Use a fresh output directory after changing protocol code;
the existing `results/protocol_v2/` artifacts were generated before the
endpoint-family bootstrap and matched no-shared baseline fixes. Preserve them
as exploratory rather than overwriting them. `results/honest/` is not modified.
Review the candidate's `deploy_metrics.json` and load-test all generated
artifacts before promoting them to the default deployment directory.
