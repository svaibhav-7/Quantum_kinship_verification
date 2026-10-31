# Corrected TSKinFace statistical audit

This audit independently recomputes paired tests from the saved fold arrays and
stores comparison tables in `tables/`. It does not modify the source results or
manuscript. Re-run it from the repository root with:

```powershell
.\.venv\Scripts\python.exe results\honest\tskinface_statistical_audit\recompute.py
```

## Executive findings

- Corrected grouped-k-fold TSKinFace baseline: ROC-AUC
  **0.7982** (fold SD
  0.0705), accuracy
  **73.98% ±
  4.03** (the runner's 1.96-SE
  interval). Historical values were ROC-AUC
  0.8814 and accuracy
  79.25%.
- The vector-minus-scalar amplitude readout effect remains positive in
  **8/8** backbone-corpus cells. Only
  **7/8** cell-level paired tests are nominally below
  0.05; corrected ArcFace TSKinFace is positive
  (+0.0610) but
  not individually significant (p=0.0643).
- Corrected composite paired-fold readout deltas (20 folds/backbone) are
  FaceNet **+0.1144**
  (95% CI [+0.0887,
  +0.1400], p=
  1.6e-08) and ArcFace
  **+0.0913**
  (95% CI [+0.0620,
  +0.1205], p=
  3e-06).
- At the more conservative corpus unit (n=4), the mean effects are
  +0.1144
  (p=0.0097) for FaceNet and
  +0.0913
  (p=0.0394) for ArcFace.
- Holm/BH counts are in `statistical_audit.json`. For the paper-compatible
  16-test family, the readout cells surviving are
  4/8
  by Holm and
  6/8
  by BH-FDR. Historically, Holm reproduces the manuscript's 5/8 readout and
  1/8 width-control counts, but standard BH-FDR recomputes to 7/8 and 3/8,
  not the manuscript's reported 6/8 and 2/8. A separate 32-test analysis
  compares against every individual
  width-matched control rather than selecting a winner.

## Historical versus corrected TSKinFace

See `tables/03_historical_vs_corrected_tskinface.csv`. The corrected
k-fold baseline is lower than the historical score. The readout gap also
shrinks: historical FaceNet and ArcFace vector-minus-scalar effects were
+0.1932
and
+0.1565;
the corrected effects are
+0.0754 and
+0.0610.

Paired historical-versus-corrected p-values in the comparison table assume
fold-index alignment. Historical files lack fold manifests and source commits,
so those comparisons are explicitly exploratory.

## Requested comparison tables

The rendered tables are in [`tables.md`](tables.md); CSVs are in [`tables/`](tables/).

1. `tables/01_scalar_vs_vector_readout.csv` — historical and current values,
   paired CIs/p-values, and multiplicity adjustments for all eight cells.
2. `tables/02_amplitude_vs_width_controls.csv` — every width-matched control,
   not only the selected best control.
3. `tables/03_historical_vs_corrected_tskinface.csv` — direct historical /
   corrected comparisons with paired-fold caveats.
4. `tables/04_standard_vqc_vs_baselines.csv` — corrected standard VQC with
   paired tests against its capacity-matched control, the alternate VQC
   readout, and the global grouped-k-fold baseline. The latter uses matched
   held-out folds but a different, pooled-corpus training protocol.
5. `tables/05_quantum_inspired_vs_classical_controls.csv` — corrected
   amplitude, standard-VQC, and set-density comparisons.
6. `tables/06_set_level_aggregation.csv` — single versus set-level results for
   each corpus and backbone, with historical and corrected labels.
7. `tables/07_dataset_and_backbone_performance.csv` — grouped-k-fold dataset
   metrics and per-backbone single/set results.
8. `tables/08_nine_formulations_scope.csv` — manuscript-reported formulations
   and whether corrected TSKinFace evidence exists.

## Interpretation and limitations

Fold tests use paired t statistics and t-based 95% intervals. Folds partition
the same corpora and are not independent population samples. The corpus-level
results are therefore shown separately. The composite substitutes corrected
TSKinFace folds while retaining historical folds for the other three corpora;
it is not a full corrected 40-fold rerun.

The historical `setlevel.json` has an implausible TSKinFace mean set size of
169.03 despite one image per identity. Its old TSKinFace set-level results are
not treated as reliable; `backbone_comparison.json` is reported separately.
Corrected density-fidelity summary artifacts omit fold scores, so no inferential
p-value is invented for those comparisons. Corrected set-level and backbone
summary means/SDs cannot be rederived from per-fold data because the runs did
not save fold-level values. All corrected amplitude and standard-VQC ROC-AUC
means/SDs do match their saved fold arrays; amplitude per-fold accuracy was
not saved. No historical TSKinFace standard VQC result was found.

The nine-formulation table in the manuscript is preserved as historical context.
Only the set-density formulation among formulations 1--9 has a corrected
TSKinFace result in the available reruns; the other formulations need their own
reruns before corrected TSKinFace claims can be made.
