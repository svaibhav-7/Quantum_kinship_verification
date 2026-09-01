# Variational Quantum Classifier for Kinship Verification — Design

**Date:** 2026-09-01
**Status:** Approved for implementation
**Origin:** Architecture proposed by project supervisor

---

## 1. What this tests, and why it is not already answered

Nine quantum-inspired formulations have been evaluated on this project against
capacity-matched classical controls. None improved on its control; four were
significantly worse. That evidence does **not** settle the architecture
proposed here, and the distinction is precise:

| | Formulations 1–9 | This design |
|---|---|---|
| Registers | two, encoded independently | **one joint 2n-qubit register** |
| Circuit | fixed Rz layer per register | **variational `U(θ)`, per-layer parameters** |
| Trainable quantum parameters | 16 | tens to hundreds |
| Cross-person interaction | only at the final fidelity | **inside the unitary** |

The earlier circuits could not express correlations *between* the two people;
they encoded each person separately and compared the results. A joint register
with entangling gates that cross the two halves can. This is a different
hypothesis with a plausible mechanism, and it is untested.

## 2. Architecture

Faithful to the supervisor's diagram:

```
X1, X2  →  φ(·)  →  z1, z2  →  |ψ_z1>, |ψ_z2>  →  U(θ)  →  measurement  →  ŷ
                                        └──── joint register ────┘
```

**Feature extraction (`φ`).** Frozen FaceNet embeddings, 512-d, as used
throughout this project. A shared encoder maps each to `n` rotation angles.
Frozen because the question here is what the circuit contributes, not what a
better representation contributes.

**Quantum encoding.** Angle encoding: `Ry(z_i)` on each qubit. Person 1
occupies qubits `0..n-1`, person 2 occupies `n..2n-1`, forming one
`2n`-qubit register of dimension `2^(2n)`.

**Variational circuit `U(θ)`.** `L` layers, each:
1. `Ry(θ) Rz(θ)` on every qubit — independent parameters per layer and qubit
2. a CNOT ring over all `2n` qubits

The ring is the load-bearing detail. It includes the bonds `n-1 → n` and
`2n-1 → 0`, which cross between the two people's halves. Without those the
circuit factorises and reduces to the already-tested case.

Parameter count: `2 × 2n × L`. At `n=4, L=3` that is 48.

**Measurement.** Two readouts, both reported:

- **Fidelity** — `|⟨ψ_ref|U(θ)|ψ⟩|²` against the all-zeros reference, a single
  scalar. The literal reading of the diagram.
- **Expectation** — Pauli-Z expectation per qubit, a `2n`-vector into a linear
  layer.

Both are implemented because the diagram says "Fidelity / Expectation", and
because the difference is itself informative: an earlier diagnosis on this
project found that compressing the decision through one scalar cost
approximately 16 ROC-AUC points. Running both separates *the circuit is
uninformative* from *the readout is lossy*.

**Decision.** Sigmoid over the readout; threshold calibrated on a
group-disjoint validation slice, as elsewhere in this project.

## 3. Sizing

The joint register costs `2^(2n)` complex amplitudes per pair.

| n | qubits | dim | state, batch 512 |
|---:|---:|---:|---:|
| 3 | 6 | 64 | 0.2 MB |
| **4** | **8** | **256** | **1.0 MB** |
| 5 | 10 | 1024 | 4.0 MB |
| 6 | 12 | 4096 | 16.0 MB |

`n=4` is the default: expressive enough for cross-register entanglement, cheap
enough to train in hours. `n` and `L` are configurable so depth and width can
be swept if the default underperforms.

## 4. The control

A classical head with the **same trainable parameter count** as `U(θ)`,
consuming the same `2n` angles, on identical folds.

This is not part of the proposed architecture and is not optional. Without it a
positive result cannot be attributed to the quantum circuit rather than to
model capacity, and a negative result cannot be distinguished from an
underpowered model. Every prior formulation on this project was evaluated this
way; this one must be too.

## 5. Components

| File | Responsibility | New? |
|---|---|---|
| `src/quantum_vqc.py` | Joint-register statevector simulator, entangling ansatz, both readouts | new |
| `src/models_vqc.py` | `VQCKinshipClassifier` and the capacity-matched control | new |
| `scripts/evaluation/run_vqc.py` | Three arms under the existing protocol | new |
| `src/quantum_fast.py` | Unchanged — it cannot express a joint register | — |
| `src/splits.py`, `src/kfold.py` | Unchanged | — |

Each is independently testable: the simulator takes angles and returns
amplitudes; the model takes embeddings and returns logits; the runner takes
folds and returns metrics.

## 6. Correctness

A variational circuit fails silently in two ways, and both are guarded:

**Wrong unitary.** The simulator is verified against an independent reference
built from explicit Kronecker products. This is not a formality: an earlier
optimisation on this project was mathematically invalid — a closed-form product
that ignored how the shared CNOT chain correlates phases — and only a
reference comparison caught it.

**No gradient flow.** A circuit whose parameters do not move trains to the
encoder's performance and looks like a null result. Tests assert that
`θ` receives non-zero gradients and that its value changes across steps.

Additional properties asserted:

- state norm is 1 after every layer
- `L=0` reduces to plain angle encoding
- output is deterministic for a fixed seed
- predictions are symmetric under swapping the two people, since kinship is a
  symmetric relation. The joint register is not symmetric by construction, so
  this is enforced by averaging both argument orders at inference.

## 7. Evaluation

Grouped 5-fold, family-disjoint, negatives regenerated per fold, thresholds
calibrated on a group-disjoint validation slice. Identical folds across arms.

Three arms:

1. **VQC-fidelity** — single-scalar readout
2. **VQC-expectation** — per-qubit Pauli-Z readout
3. **Classical control** — capacity-matched

Reported per dataset with 95% confidence intervals, plus a paired test across
folds for arms 1 and 2 against arm 3.

## 8. Success criteria

The experiment succeeds if it answers the question, whichever way it lands.

- **Positive:** a VQC arm beats the control by a margin whose confidence
  interval excludes zero. This would be the project's first supported quantum
  contribution.
- **Negative:** neither beats the control. Formulation 10 is then a full
  variational circuit with cross-register entanglement, which is a
  substantially stronger negative result than nine weaker variants.

Both outcomes are reported. The 95% CI is stated in either case, because a
difference of +0.007 with a CI spanning zero has already been rejected once on
this project and must not be reported as a gain.

## 9. Expected outcome, stated in advance

I expect the VQC will not beat the control, and record the reason so the
prediction is falsifiable: on this data the representation dominates the head.
A linear probe on frozen embeddings reaches 0.702 ROC-AUC, the trained
classical head reaches 0.810, and ArcFace embeddings alone reach 0.8616. The
gap between backbones is larger than anything any head has produced.

Recording this in advance is deliberate. If the VQC does win, the prediction
was wrong and the result is that much more interesting.

## 10. Out of scope

Fine-tuning the backbone; real quantum hardware; qubit counts beyond `n=6`
(memory); and any claim of quantum speedup — simulation is strictly more
expensive than the classical path, measured at 5–14× on this project.
