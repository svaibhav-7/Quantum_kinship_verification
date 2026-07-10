# Quantum Kinship Verification

Clean project snapshot for the latest trained architecture:

- Encoding: `entangled`
- Projection: `cross_attention`
- Qubits: `8`
- Training script: `scripts/train_hybrid.py`
- Latest run metrics: `results/training_metrics/fold_results.json`
- Ensemble checkpoints: `weights/hybrid_kinship_fold0.pt` through `weights/hybrid_kinship_fold4.pt`
- Best checkpoint: `weights/hybrid_kinship_entangled.pt`

## Structure

```text
src/
  data_loaders.py       Dataset parsing and embedding tensor preparation
  models.py             Face extractor, cross-attention projection, hybrid classifier
  quantum_core.py       Product and entangled SWAP-test simulation paths

scripts/
  train_hybrid.py       Latest 5-fold training pipeline
  predict_user_images.py  Ensemble inference for custom image pairs
  validate_quantum_circuit.py  Circuit sanity checks

weights/
  hybrid_kinship_entangled.pt
  hybrid_kinship_fold0.pt ... hybrid_kinship_fold4.pt
  embeddings_cache.pkl

results/training_metrics/
  fold_results.json
  final_evaluation_metrics.json
  training_metrics.png
  roc_curve.png
  score_distribution.png
```

Dataset folders are expected at the project root:

- `KinFaceW-I/`
- `KinFaceW-II/`
- `TSKinFace_Data/`

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the current architecture:

```bash
python scripts/train_hybrid.py --encoding-mode entangled --projection cross_attention --epochs 100
```

Run ensemble prediction on two images:

```bash
python scripts/predict_user_images.py path/to/face1.jpg path/to/face2.jpg
```

Run the sample demo:

```bash
python main.py
```

## Current Metrics

The latest cross-validation summary is stored in `results/training_metrics/fold_results.json`.

The final evaluation summary is stored in `results/training_metrics/final_evaluation_metrics.json`.
