# Set-Level and Triadic Kinship Verification — Design

**Date:** 2026-09-01
**Status:** Approved for implementation
**Supersedes:** the single-image, pairwise protocol in `RESULTS_HONEST.md` §8

---

## 1. Problem

The current pipeline reaches 75.66% accuracy / 0.8497 mean ROC-AUC under grouped
5-fold. Three constraints hold it there, all measured rather than assumed:

1. **It discards most of the data.** FIW carries 5.3 photos per identity; the
   pipeline scores one photo against one photo and throws the rest away.
2. **It ignores available structure.** TSKinFace supplies father + mother + child
   as a triad; the pipeline splits each triad into two independent pairs.
3. **One relation dominates the error.** FIW mother-son scores 0.712 AUC against
   father-daughter's 0.993, and mother-son is the largest slice (5,296 pairs).

A fourth constraint was hypothesised and **refuted**: that the frozen FaceNet
backbone caps performance. A linear probe on the same embeddings reaches only
0.702 against the trained model's 0.810, so the head earns 11 points and is not
the bottleneck.

## 2. Evidence

All figures below are family-disjoint, no training unless stated.

| Intervention | Baseline | Result | Delta |
|---|---:|---:|---:|
| **Mean-pool identity photo sets** | 0.7339 | **0.8049** | **+7.1** |
| Max over image pairs | 0.7339 | 0.7961 | +6.2 |
| **Triadic parent-pair modelling** | 0.7612 | **0.7918** | **+3.1** |
| Density fidelity `Tr(ρ₁ρ₂)` alone | 0.8050 | 0.7185 | **−8.7** |
| `Tr(ρ₁ρ₂)` added to mean-pool | 0.8050 | 0.8059 | +0.1 |
| Quantum interference sweep | 0.7684 | 0.7696 | +0.1 |

Two conclusions follow. Set-level and triadic modelling carry real signal. Both
quantum formulations do not — this is the third independent experiment to reach
that conclusion, after the 5-seed (p = 0.945) and 20-fold (p = 0.982) ablations.

`Tr(ρ₁ρ₂)` *loses* to a mean vector because it is dominated by the leading
eigenvector while accumulating noise from tail directions that cannot be
estimated from ~5 photos. The interference sweep collapses to the classical
mixture because normalised FaceNet embeddings carry no phase structure to
exploit.

## 3. Design

### 3.1 Set-level identity representation

Replace the per-image embedding with a per-identity descriptor computed from
that person's photo set `X ∈ ℝ^{N×512}`:

- `mean` — the centroid, L2-normalised (carries most of the signal)
- `spread` — mean pairwise cosine within the set (how variable the person looks)
- `n_images` — set size, so the model can learn to trust small sets less

For a pair of identities the model additionally receives the cross-set
similarity distribution: `max`, `min`, `mean`, `std` of `A·Bᵀ`. These are the
statistics that produced 0.8113 in probe 2.

**Graceful degradation is a hard requirement.** A set of size 1 must reduce
exactly to the current single-image path: `mean` is the embedding, `spread` is 0,
and the cross-set statistics all collapse to the single cosine. KinFaceW (one
photo per person) therefore keeps its present numbers, and any change there
signals a bug. This is asserted by test, not by inspection.

### 3.2 Triadic scoring

Where a complete father–mother–child triad exists, score it jointly. Features:

- `cos(F, C)` and `cos(M, C)` — the two parental similarities
- `best_mixture` — `max_α cos(normalise(αF + (1−α)M), C)` over a fixed α grid
- `α*` — the mixing weight achieving it (which parent the child resembles)
- `cos(F, M)` — parental similarity, which alone contributed +2.2 AUC

Triadic scoring is **additive, not a replacement**: pairwise scores remain the
primary output, and the triad head applies only where all three members exist.
Datasets without triads are unaffected.

### 3.3 Quantum arm as declared control

`Tr(ρ₁ρ₂)` over the set density matrix `ρ = (1/N)Σ|ψᵢ⟩⟨ψᵢ|` is retained as a
switchable feature, defaulting **off**. It is the natural quantum reading of a
photo set — a mixed state over observations — which is precisely why its failure
is informative. Reporting it as a pre-registered control across three
experiments is a stronger scientific claim than omitting it.

The existing SWAP-test branch (`use_quantum`) is likewise retained unchanged.

## 4. Components

| File | Responsibility | New? |
|---|---|---|
| `src/identity_sets.py` | Group pairs into identity photo sets; compute set descriptors | new |
| `src/set_features.py` | Pair-of-sets feature extraction, including the optional density arm | new |
| `src/triads.py` | Triad discovery and triadic features | new |
| `src/models_set.py` | `SetLevelKinshipClassifier` | new |
| `src/kfold.py` | Unchanged — folds are already grouped by family | — |
| `src/splits.py` | Unchanged | — |

Each is independently testable: set construction takes pairs and returns sets;
feature extraction takes two sets and returns a vector; the model takes feature
vectors. No component needs another's internals.

## 5. Evaluation

Unchanged protocol — grouped 5-fold, every family tested exactly once, negatives
built per side, thresholds calibrated on a group-disjoint validation slice.
**The comparison is against the current 0.8497 mean AUC under identical folds**,
so any gain is attributable to the new components rather than to a protocol
change.

Reported per dataset: accuracy ± 95% CI, ROC-AUC, and **mean set size**, so a
reader can see exactly where set-level gains originate and where the method
degrades to single-image.

Four arms, all on the same folds:

1. Single-image, pairwise (current baseline)
2. Set-level, pairwise
3. Set-level + triadic
4. Set-level + density-matrix arm (the quantum control)

Arms 2 and 3 carry the claims; arm 4 tests the quantum hypothesis a third time.

## 6. Success criteria

- Set-level ≥ +4 AUC over single-image on FIW and TSKinFace
- KinFaceW-I/II unchanged within noise (degradation correctness)
- Triadic ≥ +2 AUC on TSKinFace
- Every arm reported whatever the outcome, including if arms 2–3 disappoint

## 7. Risks

**The probe gains may not survive training.** Probes measured raw separability;
a trained head already recovers some of that signal, so the realised gain will
likely be smaller than +7.1. Mitigation: arm 1 is run on identical folds, so
the comparison stays honest even if the delta shrinks.

**Set-level changes the task.** Person-vs-person is standard in face recognition
(IJB-B/C) but not in kinship papers. Mitigation: report both protocols; the
single-image arm remains comparable to prior work.

**Mother-son may not improve.** Nothing in this design targets it directly. If
set-level lifts MS disproportionately that is a finding; if not, it stands as
stated future work rather than a silent omission.

## 8. Out of scope

Fine-tuning the FaceNet backbone (the probe shows the head is not the
bottleneck); synthetic augmentation of KinFaceW (it would manufacture the
same-photo shortcut removed from TSKinFace); and any claim that the quantum
module improves accuracy.
