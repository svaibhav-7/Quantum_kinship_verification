"""Quantum-augmented kinship classifier.

The original HybridKinshipClassifier routed every decision through a single
SWAP-test fidelity scalar. Measured on FIW that head reached 59% while plain
logistic regression on the same embeddings reached 75.1% -- the product-of-
cos^2 bottleneck was discarding signal, and gradients confirmed it (projection
MLP ~2e1, quantum params ~4e-2).

Here the fidelity becomes one input feature beside the standard symmetric pair
features. The quantum branch keeps a real gradient path and stays switchable
via `use_quantum`, so its contribution is measurable as an ablation rather
than assumed.

Symmetry: kinship is order-invariant, so pair features use only symmetric
functions (sum, |difference|, product, cosine) and the two encoders share
weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum_fast import fast_entangled_fidelity as differentiable_entangled_fidelity_vectorized


class QuantumAugmentedKinshipClassifier(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        hidden_dim=256,
        n_qubits=8,
        rel_dim=4,
        dropout=0.3,
        use_quantum=True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        self.rel_embed = nn.Sequential(
            nn.Linear(rel_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Shared encoder -> identical treatment of both faces (symmetry).
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Angle projection feeding the quantum branch.
        self.angle_proj = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_qubits),
        )
        self.ent_params1 = nn.Parameter(torch.randn(n_qubits) * 0.1)
        self.ent_params2 = nn.Parameter(torch.randn(n_qubits) * 0.1)

        # sum, |diff|, product  (all symmetric) + cosine + relation + fidelity
        feature_dim = hidden_dim * 3 + 1 + rel_dim + 1
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _encode(self, emb1, emb2, rels):
        rel_bias = self.rel_embed(rels)
        return self.encoder(emb1 + rel_bias), self.encoder(emb2 + rel_bias)

    def _angles(self, h1, h2):
        z1 = torch.tanh(self.angle_proj(h1)) * math.pi
        z2 = torch.tanh(self.angle_proj(h2)) * math.pi
        return z1, z2

    def quantum_fidelity(self, emb1, emb2, rels):
        """SWAP-test fidelity, exposed so the ablation can report it alone."""
        h1, h2 = self._encode(emb1, emb2, rels)
        z1, z2 = self._angles(h1, h2)
        fid = differentiable_entangled_fidelity_vectorized(
            z1, z2, self.ent_params1, self.ent_params2, self.n_qubits
        )
        return fid.view(-1, 1).clamp(0.0, 1.0)

    def forward_logits(self, emb1, emb2, rels):
        h1, h2 = self._encode(emb1, emb2, rels)

        # Symmetric pair features only.
        summed = h1 + h2
        absdiff = torch.abs(h1 - h2)
        product = h1 * h2
        cosine = F.cosine_similarity(h1, h2, dim=1, eps=1e-8).unsqueeze(1)

        if self.use_quantum:
            z1, z2 = self._angles(h1, h2)
            fid = differentiable_entangled_fidelity_vectorized(
                z1, z2, self.ent_params1, self.ent_params2, self.n_qubits
            ).view(-1, 1)
        else:
            # Constant keeps tensor shape identical so the ablation isolates
            # the fidelity's information, not a change in architecture.
            fid = torch.zeros(h1.shape[0], 1, device=h1.device, dtype=h1.dtype)

        feats = torch.cat([summed, absdiff, product, cosine, rels, fid], dim=1)
        return self.classifier(feats)

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))
