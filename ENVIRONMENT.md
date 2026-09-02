# Reproduction environment

Exact versions and hardware for the run that produced the JSON artefacts
in `results/honest/`. `requirements.txt` carries the looser bounds the
code needs; this file records what it was actually measured under.

## Software

| Component | Version |
|---|---|
| Python | 3.11.9 |
| OS | Windows-10-10.0.26200-SP0 |
| PyTorch | 2.11.0+cu128 |
| CUDA (torch build) | 12.8 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.9 |

## Hardware

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8.0 GiB |
| Compute capability | 12.0 |

Every experiment also runs on CPU. Absolute timings in the manuscripts
(73.5x inference, 55x training) are GPU measurements; the ratio between
arms is the reported quantity, not the wall-clock.

## Input artefacts

Embedding caches are inputs, not outputs: the evaluation scripts read
them rather than re-extracting from images, so these hashes fix the
inputs to every reported number.

| File | SHA-256 (first 16) | Size |
|---|---|---|
| `weights/caches/all_datasets_cache.pkl` | `f9a0efce3fdb6961` | 19.8 MB |
| `weights/caches/arcface_cache.pkl` | `6471a1ad8e4fec5e` | 19.8 MB |
| `weights/deploy/kinship_model.pt` | `68f28bc6050eed35` | 2.0 MB |
| `weights/deploy/set_model.pkl` | `dcc3de50f53127b0` | 0.0 MB |
| `weights/deploy/triad_model.pkl` | `14f3e6f7cfe81741` | 0.0 MB |

## Seeds and configuration

Every runner takes `--seed` (default 42) and seeds `torch`, `numpy` and
`random` per fold as `seed + fold_index`. The readout experiments were
run at the defaults below; seed sensitivity is reported over 42/101/202/303.

| Setting | Value | Flag |
|---|---|---|
| qubits per person | 6 | `--qubits` |
| circuit depth | 4 | `--depth` |
| folds | 5 (grouped, family-disjoint) | `--folds` |
| epochs | 25, early stop patience 8 | `--epochs` |
| batch size | 256 | `--batch-size` |
| learning rate | 3e-3 (AdamW, wd 1e-3) | `--lr` |
| dropout | 0.3 | `--dropout` |
| cap per family | 200 | `--cap-per-family` |
| backbone | facenet | `--backbone` |

## Regenerating each artefact

```bash
# readout ablation, all four corpora, both backbones
python scripts/evaluation/run_amplitude_vqc.py --datasets KinFaceW-I --tag amp_widthctl
python scripts/evaluation/run_amplitude_vqc.py --datasets KinFaceW-II TSKinFace FIW --tag amp_widthctl_all
python scripts/evaluation/run_amplitude_vqc.py --backbone arcface --tag amp_arcface

# negative control: every arm must fall to ~0.50
python scripts/evaluation/run_amplitude_vqc.py --datasets KinFaceW-I --shuffle-labels --tag amp_shuffled

# seed sensitivity
for s in 101 202 303; do
  python scripts/evaluation/run_amplitude_vqc.py --datasets KinFaceW-I --seed $s --tag amp_seed$s
done

# readout-width sweep and family-cap sensitivity
for k in 1 2 4 6 8 10 12; do
  python scripts/evaluation/run_amplitude_vqc.py --datasets FIW --n-observables $k --tag amp_k$k
done
for c in 50 100 200 400 800; do
  python scripts/evaluation/run_amplitude_vqc.py --datasets FIW --cap-per-family $c --tag amp_cap$c
done

# figures
python scripts/evaluation/make_figures.py
python scripts/evaluation/make_figures_readout.py
```

## Determinism

Seeds fix initialisation, fold assignment and negative sampling. GPU
kernel non-determinism means fold-level ROC-AUC can move in the fourth
decimal between runs on identical inputs; re-running the FaceNet
three-corpus job reproduced 11 of 15 fold values exactly and the rest
within 0.03. Aggregate conclusions are unaffected, but exact
bit-reproduction of a single fold is not guaranteed on different hardware.
