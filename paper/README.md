# Paper Directory

This directory contains two journal manuscripts from the Quantum Kinship
Verification project.

## Papers

| Subfolder | Format | Length | Assets | Title |
|-----------|--------|--------|--------|-------|
| `ieee/` | IEEEtran (2-col) | 12 pp | 11 figs, 13 tables, 15 refs | Person-Level Kinship Verification: Leakage-Free Evaluation, Set-Based Aggregation, and a Negative Result for Quantum-Inspired Metrics |
| `elsevier/` | elsarticle (1-col) | 24 pp | 8 figs, 14 tables, 14 refs | A Systematic Negative Result for Quantum-Inspired Similarity Metrics in Facial Kinship Verification |

Both compile with zero errors and zero undefined references. The two are
different papers from the same work: the IEEE variant leads with person-level
verification, the Elsevier variant with the quantum negative result. Page
counts are not comparable across formats --- IEEE is two-column, elsarticle is
single-column preprint style.

## Building

```bash
# IEEE paper
cd ieee && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

# Elsevier paper
cd elsevier && pdflatex -interaction=nonstopmode quantum_negative.tex && pdflatex -interaction=nonstopmode quantum_negative.tex
```

Both papers compile independently with their own figures directories.

## Data Sources

All numeric claims in both papers trace to `results/honest/*.json` in the
project root. Regenerate them with:

```bash
python scripts/evaluation/run_kfold.py            # main baseline + ablation
python scripts/evaluation/run_setlevel.py         # set-level arms
python scripts/evaluation/run_triadic.py          # triadic arms
python scripts/deploy/build_arcface_cache.py      # ArcFace embeddings
python scripts/evaluation/run_backbone_comparison.py  # cross-backbone replication
python scripts/evaluation/make_figures.py         # core figures
FIGDIR=ieee python scripts/evaluation/make_figures_extra.py   # analysis figures
FIGDIR=elsevier python scripts/evaluation/make_figures_extra.py
```

`make_figures_extra.py` also writes `results/honest/setsize_curve.json`, the
set-size sweep reported in both manuscripts.
