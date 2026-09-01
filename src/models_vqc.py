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
