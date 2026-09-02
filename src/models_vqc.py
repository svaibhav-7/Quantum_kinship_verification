"""Kinship classifiers built on the joint-register VQC, plus its control.

Three models, sharing an identical encoder so the comparison isolates what
happens after the angles are produced:

  VQCKinshipClassifier(readout="fidelity")      single scalar from U(theta)
  VQCKinshipClassifier(readout="expectation")   per-qubit <Z> vector
  CapacityMatchedControl                        classical head, same parameter
                                                count as U(theta)

The control is not optional. Without it a positive result cannot be attributed
to the quantum circuit rather than to model capacity, and a negative one cannot
be distinguished from an underpowered model.

Kinship is symmetric but a joint register is not: swapping the two people
permutes the qubits. `predict` therefore averages both argument orders, which
makes the model exactly symmetric at inference.
"""

import math

import torch
import torch.nn as nn

from .quantum_vqc import JointRegisterVQC


class _AngleEncoder(nn.Module):
    """Shared 512-d embedding -> n rotation angles in [-pi, pi]."""

    def __init__(self, embed_dim=512, hidden=128, n_qubits=4, rel_dim=4,
                 dropout=0.3):
        super().__init__()
        self.rel = nn.Sequential(
            nn.Linear(rel_dim, 32), nn.GELU(), nn.Linear(32, embed_dim))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_qubits))

    def forward(self, emb, rels):
        return torch.tanh(self.net(emb + self.rel(rels))) * math.pi


class VQCKinshipClassifier(nn.Module):
    """Encoder -> joint register -> U(theta) -> measurement -> decision."""

    def __init__(self, n_qubits_each=4, depth=3, readout="expectation",
                 embed_dim=512, hidden=128, dropout=0.3):
        super().__init__()
        if readout not in ("fidelity", "expectation"):
            raise ValueError(f"unknown readout {readout!r}")
        self.readout = readout
        self.encoder = _AngleEncoder(embed_dim, hidden, n_qubits_each,
                                     dropout=dropout)
        self.vqc = JointRegisterVQC(n_qubits_each=n_qubits_each, depth=depth)
        in_dim = 1 if readout == "fidelity" else 2 * n_qubits_each
        self.head = nn.Linear(in_dim, 1)

    def measure(self, emb1, emb2, rels):
        z1 = self.encoder(emb1, rels)
        z2 = self.encoder(emb2, rels)
        if self.readout == "fidelity":
            return self.vqc.fidelity(z1, z2)
        return self.vqc.expectations(z1, z2)

    def forward_logits(self, emb1, emb2, rels):
        return self.head(self.measure(emb1, emb2, rels))

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))

    @torch.no_grad()
    def predict(self, emb1, emb2, rels):
        """Symmetric prediction: the joint register is order-dependent, so
        both orders are averaged. Kinship is a symmetric relation."""
        return 0.5 * (self.forward(emb1, emb2, rels)
                      + self.forward(emb2, emb1, rels))

    def quantum_parameter_count(self):
        return self.vqc.theta.numel()


class CapacityMatchedControl(nn.Module):
    """Classical head with the same trainable parameter count as U(theta).

    Consumes the same angles from an identical encoder, so the only difference
    from the VQC arms is what processes them. The hidden width is solved for so
    that the head's parameter count matches `2 * 2n * depth` as closely as an
    integer allows; the achieved count is exposed for reporting.
    """

    def __init__(self, n_qubits_each=4, depth=3, embed_dim=512, hidden=128,
                 dropout=0.3):
        super().__init__()
        self.encoder = _AngleEncoder(embed_dim, hidden, n_qubits_each,
                                     dropout=dropout)
        target = 2 * (2 * n_qubits_each) * depth
        n_in = 2 * n_qubits_each

        # A Linear(n_in, w) + Linear(w, 1) costs w*(n_in + 2) + 1.
        w = max(1, round((target - 1) / (n_in + 2)))
        self.head = nn.Sequential(
            nn.Linear(n_in, w), nn.GELU(), nn.Linear(w, 1))
        self._target = target

    def head_parameter_count(self):
        return sum(p.numel() for p in self.head.parameters())

    def forward_logits(self, emb1, emb2, rels):
        z1 = self.encoder(emb1, rels)
        z2 = self.encoder(emb2, rels)
        return self.head(torch.cat([z1, z2], dim=1))

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))

    @torch.no_grad()
    def predict(self, emb1, emb2, rels):
        return 0.5 * (self.forward(emb1, emb2, rels)
                      + self.forward(emb2, emb1, rels))


class AmplitudeVQCClassifier(nn.Module):
    """The supervisor's architecture with amplitude encoding.

    Identical to `VQCKinshipClassifier` except at the input. Angle encoding
    puts `n` numbers into `n` qubits, which forced a 512-d embedding through a
    4-number bottleneck before `U(theta)` saw anything and held the model at
    chance. Amplitude encoding puts `2**n` numbers into the same `n` qubits, so
    at n=6 a 64-d projection is carried with no compression.

    The joint state is the tensor product of the two encodings, giving the full
    `2**(2n)` register. Concatenating instead yields `2**(n+1)` and silently
    builds a register of the wrong size -- a bug that occurred during
    development and is now pinned by test.

    Measured on a pilot fold: 0.6905 test ROC-AUC, against ~0.50 for the
    angle-encoded model and 0.58-0.63 for the capacity-matched control.
    """

    def __init__(self, n_qubits_each=6, depth=4, embed_dim=512, dropout=0.3,
                 readout="expectation", n_observables=None):
        super().__init__()
        if readout not in ("fidelity", "expectation"):
            raise ValueError(f"unknown readout {readout!r}")
        self.readout = readout
        self.n_each = n_qubits_each
        self.dim = 2 ** n_qubits_each
        self.proj = nn.Linear(embed_dim, self.dim)
        self.drop = nn.Dropout(dropout)
        self.vqc = JointRegisterVQC(n_qubits_each=n_qubits_each, depth=depth)

        # `n_observables` retains only the first k per-qubit <Z> values, which
        # sweeps readout width between the scalar and full-vector extremes on
        # one unchanged circuit. Qubits are ordered person-1 then person-2, so
        # small k reads only the first register -- the sweep therefore varies
        # width and cross-register coverage together, and is descriptive of the
        # width trend rather than an isolation of it.
        full = 2 * n_qubits_each
        if readout == "expectation" and n_observables is not None:
            if not 1 <= n_observables <= full:
                raise ValueError(f"n_observables must be in [1, {full}]")
            self.n_observables = n_observables
        else:
            self.n_observables = full
        self.head = nn.Linear(1 if readout == "fidelity" else self.n_observables, 1)

    def joint_state(self, emb1, emb2, rels):
        from .quantum_vqc import amplitude_encode

        a = amplitude_encode(self.drop(self.proj(emb1)), self.n_each)
        b = amplitude_encode(self.drop(self.proj(emb2)), self.n_each)
        # tensor product: (B, 2^n, 1) * (B, 1, 2^n) -> (B, 2^(2n))
        s = (a.unsqueeze(2) * b.unsqueeze(1)).reshape(a.shape[0], -1)
        return self._apply_circuit(s)

    def _apply_circuit(self, state):
        from .quantum_vqc import _apply_1q, _cnot_indices, _ry_rz

        b, dev = state.shape[0], state.device
        n = self.vqc.n_total
        for d in range(self.vqc.depth):
            for q in range(n):
                m = _ry_rz(self.vqc.theta[d, q, 0], self.vqc.theta[d, q, 1], b, dev)
                state = _apply_1q(state, m, q, n)
            for q in range(n):
                state = state[:, _cnot_indices(n, q, (q + 1) % n, dev)]
        return state

    def measure(self, emb1, emb2, rels):
        """Both branches of the diagram's "Fidelity / Expectation" box."""
        s = self.joint_state(emb1, emb2, rels)
        if self.readout == "fidelity":
            amp = s[:, 0]
            return (amp.real ** 2 + amp.imag ** 2).unsqueeze(1).clamp(0.0, 1.0)

        probs = s.real ** 2 + s.imag ** 2
        b, n = s.shape[0], self.vqc.n_total
        outs = []
        for q in range(n):
            left, right = 2 ** q, 2 ** (n - q - 1)
            p = probs.reshape(b, left, 2, right)
            outs.append(p[:, :, 0, :].sum(dim=(1, 2)) - p[:, :, 1, :].sum(dim=(1, 2)))
        return torch.stack(outs, dim=1)[:, :self.n_observables]

    def forward_logits(self, emb1, emb2, rels):
        return self.head(self.measure(emb1, emb2, rels))

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))

    @torch.no_grad()
    def predict(self, emb1, emb2, rels):
        return 0.5 * (self.forward(emb1, emb2, rels)
                      + self.forward(emb2, emb1, rels))

    def quantum_parameter_count(self):
        return self.vqc.theta.numel()


class AmplitudeControl(nn.Module):
    """Control for `AmplitudeVQCClassifier`.

    Matched on both axes that matter: the same 512 -> 2**n projection, so it
    sees identical input capacity, and a head whose parameter count matches
    U(theta). Only what processes the projected vectors differs -- a circuit in
    one case, a classical head in the other.

    Matching the projection is essential here. The amplitude VQC's advantage
    over the angle-encoded model came from carrying more input dimensions, so a
    control without the same projection would be losing on input width rather
    than on the quantum layer, and the comparison would be meaningless.
    """

    def __init__(self, n_qubits_each=6, depth=4, embed_dim=512, dropout=0.3):
        super().__init__()
        self.n_each = n_qubits_each
        self.dim = 2 ** n_qubits_each
        self.proj = nn.Linear(embed_dim, self.dim)
        self.drop = nn.Dropout(dropout)

        target = 2 * (2 * n_qubits_each) * depth   # U(theta) parameter count
        n_in = 2 * self.dim
        w = max(1, round((target - 1) / (n_in + 2)))
        self.head = nn.Sequential(
            nn.Linear(n_in, w), nn.GELU(), nn.Linear(w, 1))

    def head_parameter_count(self):
        return sum(p.numel() for p in self.head.parameters())

    def forward_logits(self, emb1, emb2, rels):
        a = self.drop(self.proj(emb1))
        b = self.drop(self.proj(emb2))
        a = a / (torch.linalg.vector_norm(a, dim=1, keepdim=True) + 1e-12)
        b = b / (torch.linalg.vector_norm(b, dim=1, keepdim=True) + 1e-12)
        return self.head(torch.cat([a, b], dim=1))

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))

    @torch.no_grad()
    def predict(self, emb1, emb2, rels):
        return 0.5 * (self.forward(emb1, emb2, rels)
                      + self.forward(emb2, emb1, rels))


class WidthMatchedControl(nn.Module):
    """Classical control matched on READOUT WIDTH, not just parameter count.

    `AmplitudeControl` matches U(theta)'s parameter count by solving for a
    hidden width, which at every configuration used in this project collapses
    to w=1 -- so it is itself a scalar-readout model. Comparing it against the
    2n-dimensional expectation readout confounds "quantum" with "not
    compressed to a scalar", which is the whole hypothesis under test.

    This control reads out exactly `2 * n_qubits_each` values from the same
    512 -> 2**n projection, matching what the circuit hands its classifier, and
    feeds them to an identical Linear(2n, 1). `mode` selects what produces
    those values:

      "linear" : a learned linear map              (equal-width projection)
      "mlp"    : a learned nonlinear map           (equal-width MLP)
      "random" : a fixed random projection         (no learning in the readout)

    Together with the quantum arm these separate three explanations of the
    observed margin: quantum structure, readout width, or learned nonlinearity.
    """

    def __init__(self, n_qubits_each=6, depth=4, embed_dim=512, dropout=0.3,
                 mode="linear", hidden=64):
        super().__init__()
        if mode not in ("linear", "mlp", "random"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.n_each = n_qubits_each
        self.dim = 2 ** n_qubits_each
        self.width = 2 * n_qubits_each          # == the circuit's readout width
        self.proj = nn.Linear(embed_dim, self.dim)
        self.drop = nn.Dropout(dropout)

        n_in = 2 * self.dim
        if mode == "linear":
            self.readout = nn.Linear(n_in, self.width)
        elif mode == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, self.width))
        else:
            self.readout = nn.Linear(n_in, self.width)
            for p in self.readout.parameters():
                p.requires_grad_(False)
        self.head = nn.Linear(self.width, 1)     # identical to the VQC's head

    def head_parameter_count(self):
        return sum(p.numel() for p in self.head.parameters()) + sum(
            p.numel() for p in self.readout.parameters() if p.requires_grad)

    def _features(self, emb1, emb2):
        a = self.drop(self.proj(emb1))
        b = self.drop(self.proj(emb2))
        a = a / (torch.linalg.vector_norm(a, dim=1, keepdim=True) + 1e-12)
        b = b / (torch.linalg.vector_norm(b, dim=1, keepdim=True) + 1e-12)
        return torch.cat([a, b], dim=1)

    def forward_logits(self, emb1, emb2, rels):
        return self.head(torch.tanh(self.readout(self._features(emb1, emb2))))

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))

    @torch.no_grad()
    def predict(self, emb1, emb2, rels):
        return 0.5 * (self.forward(emb1, emb2, rels)
                      + self.forward(emb2, emb1, rels))
