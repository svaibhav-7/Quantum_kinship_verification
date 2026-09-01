# Paper Directory

This directory contains two journal manuscripts from the Quantum Kinship
Verification project.

## Papers

| Subfolder | Format | Title | Status |
|-----------|--------|-------|--------|
| `ieee/` | IEEEtran | Person-Level Kinship Verification: Leakage-Free Evaluation, Set-Based Aggregation, and a Negative Result for Quantum-Inspired Metrics | Draft |
| `elsevier/` | elsarticle | A Systematic Negative Result for Quantum-Inspired Similarity Metrics in Facial Kinship Verification | Draft |

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
project root and are reproducible via `scripts/evaluation/run_kfold.py`.
