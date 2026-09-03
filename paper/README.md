# Paper Directory

This directory contains two journal manuscripts from the Quantum Kinship
Verification project, plus a second typesetting of one of them.

## Papers

| Subfolder | Format | Length | Assets | Title |
|-----------|--------|--------|--------|-------|
| `ieee/` | IEEEtran (2-col) | 13 pp | 11 figs, 13 tables, 15 refs | Person-Level Kinship Verification: Leakage-Free Evaluation, Set-Based Aggregation, and a Negative Result for Quantum-Inspired Metrics |
| `elsevier/` | elsarticle (1-col) | 35 pp | 11 figs, 20 tables, 15 refs | Beyond Scalar Fidelity: Readout Bottlenecks in Quantum-Inspired Facial Kinship Verification |

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

## The third file: an Elsevier edition of the person-level paper

`elsevier_person/` is the same manuscript as `ieee/`, typeset for an Elsevier
(elsarticle) submission rather than IEEEtran. It is **generated**, not
maintained:

```bash
python scripts/convert_ieee_to_elsarticle.py
```

Edit `ieee/main.tex`; never `elsevier_person/main.tex`, which is overwritten.
Only format-bearing constructs are rewritten -- document class, frontmatter,
keyword environment, and the column-width commands, since `\columnwidth` is
half a page under two-column IEEEtran but a full page under single-column
elsarticle. `tests/test_manuscripts.py` asserts that the two bodies differ in
nothing but those layout commands and that their numeric claims are identical,
so the editions cannot drift.

Submit one or the other, not both: they are the same paper.

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
