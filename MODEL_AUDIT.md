1# Implementation and Research Audit

Date: 2026-07-01

## Pipeline Status

The active implementation is centered on:

- `src/models.py`: FaceNet feature extraction, relation-conditioned cross-attention projection, and `HybridKinshipClassifier`.
- `src/quantum_core.py`: product SWAP-test circuits, shared CNOT/Rz circuit variant, Qiskit verification, and differentiable PyTorch fidelity.
- `src/data_loaders.py`: KinFaceW-I/II and TSKinFace pair parsing plus embedding caching.
- `scripts/train_hybrid.py`: 5-fold cross-validation, Youden thresholding, fold checkpoint saving, and ensemble evaluation.
- `scripts/predict_user_images.py`: user-image ensemble inference using fold checkpoints and saved optimal thresholds.
- `scripts/test_real_time_pairs.py`: visual/full-set evaluation.
- `scripts/test_pipeline_timing.py`: timing and Qiskit-vs-forward benchmarking.
- `scripts/generate_publication_metrics.py`: metric tables and plot bundle generation.

These files should be kept because they are part of the updated cross-attention and cross-validation workflow.

## Architecture Assessment

The cross-attention projection is relation-conditioned and trainable, but each face embedding is treated as a single token. That means the attention layer learns cross-conditioned embedding transformations; it is not token-level attention over facial regions or embedding subfeatures.

The current `entangled` mode applies a CNOT chain and shared Rz phases to both compared states before measuring fidelity. Since fidelity is invariant under applying the same unitary to both states, the implemented shared-circuit fidelity is equivalent to the product-state formula for the same projected angles:

```text
max_abs_diff(differentiable_entangled_fidelity, analytical_product_fidelity)
= 1.49e-08 on a random 5-pair local check
```

So the present implementation should not be described as proving a non-classically-simulable fidelity advantage. It is still a valid circuit-verified SWAP-test pipeline and a useful trained projection system, but the "entanglement gain" needs a redesigned circuit or input-dependent/non-shared unitary before it can support that research claim.

## Metric Assessment

Stored result files report:

- Final Qiskit verification: accuracy 79.83%, ROC-AUC 0.8604, F1 80.26%, precision 78.60%, recall 81.99%, MCC 0.5972.
- 5-fold cross-validation mean optimal accuracy: 64.26% +/- 1.82%.
- Ensemble optimal accuracy: 76.17%, ROC-AUC 0.8325.

Research interpretation: the final metric is promising for a small KinFaceW-I benchmark, especially compared with the stored product baseline, but the gap between fold-level CV and final-test reporting is large. The strongest defensible claim is that the trained cross-attention/SWAP-test pipeline can produce competitive held-out results, while model stability and leakage-free evaluation need careful discussion.

## Cleanup Decision

Kept:

- Core source, training/evaluation/prediction scripts, trained fold checkpoints, metric JSON, plots, datasets, paper, README, and assets.

Removed or safe to remove:

- Python bytecode caches (`__pycache__`).
- Local editor settings (`.vscode`) because they are environment-specific and not part of the research pipeline.
- Stale per-sample `results/real_time_test/pair_*.png` images are already absent from the workspace and remain deleted.
