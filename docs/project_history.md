## Project History: How This Ended Up Here

This repository began as a quantum-inspired kinship verification system. It is
now a leakage-free evaluation protocol with a documented negative result for
quantum-inspired metrics. That change was driven by measurements, and the route
is recorded here because the wrong turns are as informative as the destination.

### Where it started

The original system mapped FaceNet embeddings into an 8-qubit simulated Hilbert
space, compared them by SWAP-test fidelity, and combined eleven sub-models into
a meta-ensemble. It reported **77.8% on FIW** and **83.3% on TSKinFace**, and
claimed that removing the quantum module cost 7.5% accuracy.

The trigger for re-examination was mundane: most datasets sat near 60% while
TSKinFace alone crossed 75%. That asymmetry turned out to matter more than it
looked.

### Problem 1 — a trivial baseline beat the whole pipeline

The first diagnostic compared the quantum pipeline against ordinary classifiers
on **identical embeddings and identical pairs**:

| Method | Accuracy | ROC-AUC |
|---|---:|---:|
| Raw cosine similarity, no training | 68.0% | 0.734 |
| Logistic regression on `[abs_diff, product, cosine]` | **75.1%** | **0.821** |
| The quantum meta-ensemble | 59.0% | 0.591 |

A twenty-line logistic regression already cleared the 75% target and beat the
full quantum pipeline by 16 points. A shuffled-label control returned 45.4%,
confirming the 75.1% was real signal rather than leakage in the probe.

### Problem 2 — every number was measured with the answer key in training

FIW draws 57,120 pairs from only **95 families and 472 identities**, so each
identity recurs roughly 242 times. Under the pair-level random splitting the
code used:

```
test pairs sharing an identity with train : 11,424/11,424 = 100.0%
```

Not most of them. All of them. Every metric the project had produced was
optimistic by an unknown margin.

Fixing this was less direct than expected. Splitting on families alone fails,
because **every non-kin pair joins two families**: treating families as vertices
and negatives as edges yields a graph with a single connected component
spanning all 95 families. No family-disjoint partition of the existing pairs
exists. Negatives must be discarded and regenerated inside each side after the
split. A splitter that passed its synthetic unit tests failed on the real corpus
for exactly this reason — which is why the test suite now verifies on real data,
not only on fixtures.

### Problem 3 — TSKinFace's 83% measured the wrong thing

```
KIN pairs from SAME family:     800/800 = 100.0%
NON-KIN pairs from SAME family:   0/800 =   0.0%
```

Family membership predicted the label perfectly, so a model could score highly
by recognising a shared photo session — lighting, camera, background — rather
than kinship. Rebuilding the negatives so half are drawn *within* a family
(father vs mother: co-photographed, not kin) drops raw separability from 0.7278
to 0.6880. **That ~9 AUC gap was the shortcut.**

Prior work has reported related construction artefacts in kinship corpora, so
this is a confirmation rather than a discovery, and the manuscripts say so.

### Problem 4 — single splits were too unstable to report

Even after fixing the leak, a single group-disjoint split gave accuracies
ranging from **61.7% to 84.1%** depending only on the random seed, because FIW's
family sizes are severely skewed (one family holds 19% of all pairs). Grouped
5-fold, with every family tested exactly once and a per-family cap, reduced the
spread from 22.5 points to 12.1.

---

## Why the Quantum Module Did Not Work

Nine distinct quantum-inspired formulations were tested, each against a
classical control matched for capacity or intent. They span the three roles a
quantum component could plausibly play.

| # | Formulation | Role | Control | Result |
|---|---|---|---|---|
| 1 | SWAP-test fidelity (5 seeds) | similarity | classical head | p = 0.945 |
| 2 | SWAP-test fidelity (20 folds) | similarity | classical head | p = 0.982 |
| 3 | Density fidelity `Tr(rho1 rho2)` | similarity | mean-pooling | **-8.7 AUC** |
| 4 | Interference phase sweep | similarity | classical mixture | +0.1 AUC |
| 5 | Entanglement entropy | feature | cosine | 0.514 (chance) |
| 6 | POVM measurement head | inductive bias | capacity-matched head | **-5.5 AUC** |
| 7 | Purity regulariser | inductive bias | decorrelation penalty | **-0.9, p = 0.047** |
| 8 | Density arm on photo sets | similarity | set features | -0.0006 |
| 9 | Interference sweep on triads | similarity | convex mixture | -0.0006 |

**None improved on its control; four were significantly worse.** Formulations
8 and 9 were additionally re-tested under a second backbone and failed on all
eight backbone-corpus combinations.

### The causes are structural, not implementational

**Interference formulations collapse to their classical counterparts.**
L2-normalised FaceNet embeddings carry no phase structure to exploit. The phase
sweep in formulation 4 is maximised at phase zero — which is precisely the
classical convex mixture.

**Density-matrix formulations cannot be estimated.** A `d x d` operator built
from a median of five observations is dominated by its leading eigenvector,
which is simply the mean, while accumulating noise from tail directions that
are not estimable at that sample size. This is why `Tr(rho1 rho2)` scores
*below* plain mean-pooling rather than above it.

**The SWAP-test fidelity was an information bottleneck.** Routing every decision
through one scalar discarded signal the classifier could otherwise use. Gradient
norms confirmed it: the projection MLP received ~2e1 while the quantum
parameters received ~4e-2.

### There is no speedup either

A frequent assumption is that the quantum path is at least faster. It is not.
Measured on an RTX 5060, the quantum branch is **5-14x slower** than the
classical path at every qubit count from 2 to 12, and the gap widens with qubit
count because the statevector grows as `2^n`. Simulating a circuit is strictly
more work than the classical operation it parallels.

The 380x speedup this project does report is **quantum versus quantum** — an
optimised simulator against the naive one — not quantum versus classical. That
distinction is easy to lose and is stated explicitly in both manuscripts.

### What the quantum work is worth

Two things, both genuine:

1. **An exact simulator optimisation.** The reference implementation
   materialised a `2^n` statevector and permuted all `n` axes once per qubit,
   leaving the GPU idle on kernel-launch overhead. Applying rotations by
   reshaping, and precomputing the CNOT chain and Rz layer as a single diagonal
   phase vector, gives **192 -> 73,094 pairs/s** with identical output and
   gradients, verified by test. A closed-form product over qubits is *not*
   available here: the shared CNOT chain correlates the Rz phases, so the
   overlap does not factorise.

2. **A rigorous negative result.** Nine pre-registered formulations against
   matched controls is more evidence than most positive claims in this area
   carry.

---

## The New Methodology

### 1. Leakage-free evaluation protocol

- **Family-disjoint splitting** (`src/splits.py`) with negatives regenerated
  per side, since pre-existing negatives make the family graph connected.
- **Grouped k-fold** (`src/kfold.py`): every family tested exactly once,
  dealt largest-first into the smallest fold to keep folds balanced despite
  skewed family sizes.
- **Per-family capping**: no family may contribute more than 1,500 pairs, so no
  single family can dominate a fold.
- **Dataset-scoped group keys**: `fd_003` names *different* families in
  KinFaceW-I and KinFaceW-II; an unscoped key silently merges them.
- **Shortcut-free TSKinFace negatives** (`src/ts_pairs.py`).
- **FPR-constrained threshold calibration** (`src/calibration.py`): Youden's J
  produced a 24.3% false-positive rate, which is the wrong trade for
  verification.

Fold integrity is verified directly rather than assumed: zero folds with
train/test group overlap, every family tested exactly once, no duplicates
(95/95 FIW, 533/533 KinFaceW-I, 1000/1000 KinFaceW-II, 559/559 TSKinFace).

### 2. Person-level rather than photograph-level verification

The conventional protocol scores one photograph against one photograph, but FIW
supplies a **median of five photographs per identity** — so the standard
formulation discards most of the available evidence about each person.

`src/identity_sets.py` represents each *person* by their photo set: normalised
centroid, within-set spread, set size, and the max/min/mean/std of the cross-set
similarity matrix.

**Graceful degradation is a hard requirement, not a nicety.** A set of size one
must reduce *exactly* to the single-image path, which is why the three
single-photograph corpora are bit-identical to the baseline. This is asserted by
test, not by inspection.

### 3. Triadic scoring

Where father, mother and child are all available, the triad is scored jointly
(`src/triads.py`) rather than decomposed into two independent pairs. The
strongest contributor is `cos(F, M)` — parental similarity — which no pairwise
model can express: a child resembling one parent is weaker evidence of kinship
when the parents already resemble each other.

### 4. Deliberately conventional architecture

Frozen FaceNet backbone, shared Siamese encoder, symmetric pair features,
logistic regression. **497,881 parameters, every block textbook.** This is a
design choice, not an oversight: the protocol findings cannot be attributed to
architectural novelty if the architecture has none. The papers state this
explicitly.

---

## Results Under the New Protocol

**Baseline**, grouped 5-fold, every family tested exactly once:

| Dataset | Accuracy (95% CI) | ROC-AUC |
|---|---:|---:|
| KinFaceW-I | 76.92% +/- 2.19 | 0.8752 |
| KinFaceW-II | 74.25% +/- 1.95 | 0.8320 |
| FIW | 72.21% +/- 1.55 | 0.8101 |
| TSKinFace (shortcut-free) | 79.25% +/- 2.32 | 0.8814 |
| **Mean** | **75.66%** | **0.8497** |

**Person-level aggregation**, and its replication under a second backbone:

| Corpus | Photos/id | FaceNet single -> set | ArcFace single -> set |
|---|---:|---|---|
| FIW | 7.30 | 0.7491 -> 0.8254 (**+0.0763**) | 0.7764 -> 0.8740 (**+0.0976**) |
| KinFaceW-I | 1.00 | identical | identical |
| KinFaceW-II | 1.00 | identical | identical |
| TSKinFace | 1.00 | identical | identical |

The gain is **larger** under the stronger backbone, consistent with the proposed
mechanism: aggregation suppresses per-photograph nuisance variation, and a
better embedding has more signal left to recover once it is suppressed.

**How the gain scales with evidence supplied** (FIW, no training):

| Photographs | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.7339 | 0.7744 | 0.7951 | 0.8042 |
| Gain | — | +0.041 | +0.061 | +0.070 |

A second photograph alone is worth over half the total available gain, and the
curve is largely flat beyond five — a practical answer to how many photographs
are worth collecting.

**Triadic scoring** (TSKinFace): 0.7569 -> **0.7893** (+0.0324, p = 0.0001,
positive in all five folds).

---

## A Correction Worth Recording

An earlier version of this work reported that mother-son pairs were markedly
harder than other relations (0.712 ROC-AUC against father-daughter's 0.993) and
attributed the gap to training dynamics. **That claim was wrong**, and how it
failed is instructive.

Investigating a fix produced three findings in sequence:

1. Training on mother-son alone gave **0.7989 against 0.8005** for the shared
   model — head-sharing was not the cause, so a per-relation head would not
   have helped.
2. The test fold in question drew **92% of its mother-son pairs from a single
   family**, which scores 0.701. The only other contributing family scores
   0.846.
3. Under grouped k-fold with a per-family cap, mother-son reaches **0.7910 —
   second best of four** — and the four relations span 0.031, close to the
   0.078 measured on raw embeddings.

**The relation effect was a family effect.** The methodological lesson
generalises: leakage-free splitting is *necessary but not sufficient* on a
corpus where one family holds 19% of the pairs. A subgroup difference observed
on a single split of a skewed corpus should be treated as provisional until it
survives resampling.

---

## What This Work Does Not Claim

Stated plainly, because each is checkable in minutes:

- **No architectural novelty.** The model is conventional by design.
- **No quantum advantage.** Nine formulations, none beating its control.
- **No quantum speedup.** The quantum path is 5-14x *slower* than classical.
- **Not state of the art.** 75.66% mean accuracy sits below published figures
  of roughly 80-82%. Those figures are obtained under protocols admitting the
  leakage quantified here, so the comparison is not like-for-like — but that is
  an argument, not a result, and the papers present it as such.
- **Set-level gains are confined to one corpus.** Only FIW stores multiple
  photographs per identity; the effect is exactly zero on the other three, by
  construction.

---

## Reproducing Everything

```bash
python scripts/deploy/build_all_caches.py             # FaceNet embeddings
python scripts/deploy/build_arcface_cache.py          # ArcFace embeddings
python scripts/evaluation/run_kfold.py                # baseline + quantum ablation
python scripts/evaluation/run_kfold.py --no-quantum --tag kfold_noq
python scripts/evaluation/run_setlevel.py             # person-level arms
python scripts/evaluation/run_triadic.py              # triadic arms
python scripts/evaluation/run_backbone_comparison.py  # cross-backbone replication
python scripts/evaluation/make_figures.py             # core figures
FIGDIR=ieee python scripts/evaluation/make_figures_extra.py
pytest tests/ -q                                      # 197 tests
```

Every numeric claim in this README and in both manuscripts derives from a JSON
artefact in `results/honest/`. Each experiment is deterministic given its seed:
repeated runs reproduce the reported figures to four decimal places.

---

## Manuscripts

| Path | Format | Length | Lead contribution |
|---|---|---|---|
| `paper/ieee/main.tex` | IEEEtran | 12 pp | Person-level verification |
| `paper/elsevier/quantum_negative.tex` | elsarticle | 24 pp | The quantum negative result |

Both compile with zero errors and zero undefined references. They are different
papers from the same work; see `paper/README.md`.

---

