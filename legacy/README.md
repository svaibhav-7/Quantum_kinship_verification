# Legacy — superseded material

Everything in this directory is **superseded and should not be cited**. It is
kept for provenance only.

## Why it was archived

### `research_experiments/` — the 12-module suite

Every module here was built on the previous architecture and evaluated with
**pair-level random splits**, under which 100% of test pairs shared an identity
with the training set (measured: 11,424/11,424). Its outputs are therefore
optimistic and internally inconsistent. For example
`outputs/01_ablation_study/ablation_results.json` reports the full model at
51.3% on KinFaceW-I while "− Quantum Module" reaches 67.3% — i.e. the ablation
that removes the quantum module *beats* the full model, contradicting the
README claim that removing it costs 7.5%.

`02_sota_literature_comparison/run_sota_comparison.py` also hardcodes
comparison baselines (HAN-Kin, MTKT, CKG, …) for which no citation or source
could be found in this repository. They must not be reused.

### `paper/main.tex`

Reports the pre-correction figures (77.8% / 76.0% / 0.860) and reproduces the
unsourced baselines above. Two of them (MTKT, CKG) carry no `\cite{}` at all,
and HAN-Kin cites a 2021 reference for a paper the table dates to 2023.
Any future manuscript should start from `RESULTS_HONEST.md`.

## What replaces it

| Superseded | Replacement |
|---|---|
| 12-module suite | `scripts/evaluation/run_kfold.py` (grouped 5-fold) |
| Random pair splits | `src/splits.py`, `src/kfold.py` |
| Stock TSKinFace pairs | `src/ts_pairs.py` (shortcut-free negatives) |
| Meta-ensemble | `src/models_hybrid.py` |
| Hardcoded accuracies | Measured metrics in `results/honest/` |

Current, verified numbers live in `RESULTS_HONEST.md`.
