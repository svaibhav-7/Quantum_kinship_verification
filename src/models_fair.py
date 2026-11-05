"""Reference model for protocol v2: a cosine prior plus a learned projection.

`QuantumAugmentedKinshipClassifier` adds a learned relation vector to each
unit-norm embedding and also feeds the relation to its classifier. On negatives
that match the positives' relation, generation and gender, that path held
validation ROC-AUC at 0.507; removing it alone gave 0.760 (fold 1 training side,
two seeds). The same path also produced the dead seed in the corrected k-fold.

This model takes no relation input. Untrained it scores close to plain cosine,
and it reached 0.769 on the same validation data with far less overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_hybrid import QuantumAugmentedKinshipClassifier


class FairKinshipModel(nn.Module):
    def __init__(self, embed_dim=512, proj_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.cos_weight = nn.Parameter(torch.tensor(5.0))
        self.proj_weight = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward_logits(self, emb1, emb2):
        a, b = F.normalize(emb1, dim=1), F.normalize(emb2, dim=1)
        raw = F.cosine_similarity(a, b, dim=1)
        learned = F.cosine_similarity(
            self.proj(self.drop(a)), self.proj(self.drop(b)), dim=1)
        return (self.cos_weight * raw + self.proj_weight * learned
                + self.bias).unsqueeze(1)

    def forward(self, emb1, emb2):
        return torch.sigmoid(self.forward_logits(emb1, emb2))


class RelationFreeHybrid(QuantumAugmentedKinshipClassifier):
    """The SWAP-branch host model with its relation path removed.

    Zeroing the relation input is not enough: `rel_embed` still adds a learned
    constant vector to each unit-norm embedding, and training stayed near
    chance (0.56 on RFIW). Quantum on/off comparisons need a host model that
    can learn, so this one encodes the face directly and ignores relations.
    """

    def _encode(self, emb1, emb2, rels):
        return self.encoder(emb1), self.encoder(emb2)

    def forward_logits(self, emb1, emb2, rels):
        return super().forward_logits(emb1, emb2, torch.zeros_like(rels))


class ScalarReadoutModel(nn.Module):
    """One-number readouts that give "a scalar" a fair chance.

    The scalar arm of the readout study is |<0...0|U|a,b>|^2. Its values sit
    near 2^-2n, far below the scale of the <Z> vector, and it does not compare
    the two people. These arms test whether one number as such is the
    limitation, or only that particular number:

      fidelity_log : the same circuit probability, log-scaled and standardised
      swap         : |<a|b>|^2 on the same projection, which does compare them
    """

    KINDS = ("fidelity_log", "swap")

    def __init__(self, kind, n_qubits_each=6, depth=4, embed_dim=512, dropout=0.3):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {self.KINDS}")
        from .models_vqc import AmplitudeVQCClassifier

        self.kind = kind
        self.vqc = AmplitudeVQCClassifier(
            n_qubits_each=n_qubits_each, depth=depth, embed_dim=embed_dim,
            dropout=dropout, readout="fidelity")
        self.proj = self.vqc.proj
        self.drop = self.vqc.drop
        self.norm = nn.BatchNorm1d(1)
        self.head = nn.Linear(1, 1)

    def raw_scalar(self, emb1, emb2):
        if self.kind == "swap":
            a = F.normalize(self.drop(self.proj(emb1)), dim=1)
            b = F.normalize(self.drop(self.proj(emb2)), dim=1)
            return (a * b).sum(dim=1, keepdim=True) ** 2
        return self.vqc.measure(emb1, emb2, None)

    def feature(self, emb1, emb2):
        value = self.raw_scalar(emb1, emb2)
        if self.kind == "fidelity_log":
            value = torch.log(value.clamp_min(1e-12))
        return self.norm(value)

    def forward_logits(self, emb1, emb2, rels=None):
        return self.head(self.feature(emb1, emb2))

    def forward(self, emb1, emb2, rels=None):
        return torch.sigmoid(self.forward_logits(emb1, emb2))

    @torch.no_grad()
    def predict(self, emb1, emb2, rels=None):
        return 0.5 * (self.forward(emb1, emb2) + self.forward(emb2, emb1))
