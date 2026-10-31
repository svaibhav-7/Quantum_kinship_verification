# TSKinFace negative-pair protocol

## Why this exists

TSKinFace positive parent-child pairs are within-family. The shortcut-free
protocol also requires non-kin father-mother pairs from the same family:
these pairs share the family/photo-session context without being parent-child.
Cross-family parent-child negatives are retained as a second negative regime.

The evaluation runners previously filtered the output of
`build_tskinface_pairs(...)` down to positives before splitting. Their generic
negative samplers then generated only cross-family negatives, so the resulting
evaluation no longer contained the intended father-mother examples.

## Corrected protocol

1. Build the positive parent-child pairs.
2. Split positives into group-disjoint fit, validation, and test subsets.
3. For each subset independently, call
   `src.ts_pairs.add_tskinface_negatives`.
4. The helper reconstructs one father-mother negative for every family with
   both parent endpoints available in that subset. It samples additional
   parent-child negatives only across families represented in that same subset.
5. No cross-family negative is allowed to bridge two partitions.

The helper rejects non-positive input and positive pairs that cross family
boundaries. It is used by the active TSKinFace paths in:

- `scripts/evaluation/run_kfold.py`
- `scripts/evaluation/run_vqc.py`
- `scripts/evaluation/run_amplitude_vqc.py`
- `scripts/evaluation/run_setlevel.py`
- `scripts/evaluation/run_backbone_comparison.py`

The `--cap-per-family` CLI option in these runners is applied to FIW
positive pairs; it is **not** a TSKinFace family cap. Each result JSON now
records the source commit, CLI arguments, TSKinFace pair/family counts after
embedding coverage, the negative-pair policy, and the cap status.

## Regression tests

Run the isolated protocol tests with:

```bash
python -m pytest tests/test_tskinface_negatives.py -q
```

The tests include synthetic FMS/FMD family trees, same-family father-mother
negative checks, cross-family sampling checks, and invalid-input checks. The
dataset-backed checks run when TSKinFace data is present and otherwise skip.

## Re-running corrected experiments

Use new tags so the old result artifacts remain available for comparison:

```bash
python scripts/evaluation/run_kfold.py --tag kfold_tskinface_corrected
python scripts/evaluation/run_vqc.py --datasets TSKinFace --tag vqc_tskinface_corrected
python scripts/evaluation/run_amplitude_vqc.py --datasets TSKinFace --backbone facenet --tag amplitude_vqc_tskinface_facenet_corrected
python scripts/evaluation/run_amplitude_vqc.py --datasets TSKinFace --backbone arcface --tag amplitude_vqc_tskinface_arcface_corrected
python scripts/evaluation/run_setlevel.py --tag setlevel_tskinface_corrected
python scripts/evaluation/run_backbone_comparison.py --tag backbone_comparison_tskinface_corrected
```

The `run_kfold`, `run_setlevel`, and `run_backbone_comparison` commands
also evaluate the other datasets; those scripts do not currently expose a
dataset-only filter. The corrected results must replace the old TSKinFace
values in pooled summaries and statistical tests before manuscript claims
are updated.
