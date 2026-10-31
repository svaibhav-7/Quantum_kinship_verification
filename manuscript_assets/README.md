# Corrected manuscript tables and figures

These are **replacement assets** for the corrected-audit manuscript, generated from the repository audit at `results/tskinface-corrected-2026-10-31`.

## Contents
- `figures/fig_baseline_corrected.tex`: corrected grouped five-fold baseline (replaces stale Figure 7).
- `figures/fig_readout_corrected.tex`: scalar-versus-vector readout results (replaces stale readout figure on manuscript page 31).
- `tables/tab_baseline_corrected.tex`: Table 4 values.
- `tables/tab_readout_corrected.tex`: Table 16 values, with historical/corrected provenance.
- `tables/tab_tskinface_width_controls.tex`: corrected TSKinFace individual width-control comparisons.
- `data/*.csv`: compact source data for reproducibility.

## Overleaf use
Upload the `.tex` snippets and add `\usepackage{pgfplots}`, `\pgfplotsset{compat=1.18}`, `\usepackage{booktabs}`, and `\usepackage{threeparttable}` (if using notes). Insert a figure with `\input{figures/fig_baseline_corrected.tex}` or `\input{figures/fig_readout_corrected.tex}`; insert tables with `\input{tables/<filename>.tex}`.

The figure snippets use PGFPlots and are vector-rendered during Overleaf compilation; they are not raster screenshots.

## Provenance and interpretation
- Corrected grouped-k-fold summaries: `kfold_tskinface_corrected.json` (audit commit source `a480275269855e3ecffe7b3f922793e9b57d97e5`).
- Readout and width-control comparisons: `01_scalar_vs_vector_readout.csv` and `02_amplitude_vs_width_controls.csv`.
- Only TSKinFace rows are corrected in the multi-corpus readout composite. FIW, KinFaceW-I and KinFaceW-II remain historical results. The figure and table explicitly disclose this.
- Fold-level intervals and p-values are descriptive because folds within a corpus partition the same identity pool.
- The old Figure 8 entropy bar cannot be responsibly regenerated from the currently available summary table: its plotted -22.24 pp value is not reconciled with Table 7's “0.514 alone” entry. It is intentionally not replaced here pending the raw entropy experiment artifact.
