"""
Models for Quantum-Classical Hybrid Kinship Verification.

Contains:
  - FaceFeatureExtractor: Pretrained CNN (FaceNet / ResNet-18) for 512-d embeddings
  - CrossAttentionProjection: Multi-head cross-attention + MLP projection (new)
  - HybridKinshipClassifier: Main classifier with product / shared-circuit modes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import warnings
import math

from .quantum_core import (
    simulate_swap_test_batch,
    simulate_entangled_swap_test_batch,
    analytical_product_fidelity,
    differentiable_entangled_fidelity,
)

# =============================================================================
# 1. FACE FEATURE EXTRACTOR (Unchanged)
# =============================================================================


class FaceFeatureExtractor:
    """
    Classical CNN Face Feature Extractor.
    Extracts high-quality facial embeddings using FaceNet (InceptionResnetV1) pretrained on VGGFace2.
    Gracefully falls back to torchvision ResNet-18 if FaceNet fails or is unavailable.
    """

    def __init__(self, use_resnet_fallback=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fallback = use_resnet_fallback

        # Define standard image transform (resize, convert to tensor, normalize)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if not self.fallback:
            try:
                from facenet_pytorch import InceptionResnetV1

                print(
                    "Initializing FaceNet (InceptionResnetV1) pretrained on VGGFace2..."
                )
                self.model = (
                    InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
                )
                self.embedding_dim = 512
                print("FaceNet feature extractor successfully initialized.")
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize FaceNet ({e}). Falling back to torchvision ResNet-18."
                )
                self.fallback = True

        if self.fallback:
            print("Initializing torchvision ResNet-18 pretrained on ImageNet...")
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(
                self.device
            )
            self.model.fc = (
                nn.Identity()
            )  # Remove classifier to output 512-dim features
            self.model.eval()
            self.embedding_dim = 512
            print("ResNet-18 feature extractor successfully initialized.")

    @torch.no_grad()
    def extract(self, img_path):
        """
        Extracts L2-normalized 512-dimensional embedding for a single face image.
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        emb = self.model(tensor).squeeze()
        emb_np = emb.cpu().numpy()
        norm = np.linalg.norm(emb_np) + 1e-8
        return emb_np / norm

    @torch.no_grad()
    def extract_batch(self, img_paths):
        """
        Extracts L2-normalized embeddings for a batch of face images.
        """
        tensors = []
        for path in img_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path).convert("RGB")
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)
        embs = self.model(batch)
        embs_np = embs.cpu().numpy()
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-8
        return embs_np / norms


# =============================================================================
# 2. CROSS-ATTENTION PROJECTION NETWORK (New)
# =============================================================================


class CrossAttentionProjection(nn.Module):
    """
    Projects face embeddings into quantum angle parameters using
    multi-head cross-attention between the two face representations.

    Architecture:
      1. Each embedding (512-d) is attended to by the other via cross-attention
      2. Residual connection + LayerNorm
      3. MLP projection: 512 → 256 → 128 → n_qubits
      4. tanh scaling to [-π, π]

    Note: each embedding is represented as a single token, so the attention
    operation learns cross-conditioned projections rather than token-level
    facial-region alignment.
    """

    def __init__(self, embed_dim=512, n_qubits=8, n_heads=8, rel_dim=4, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits

        # Relation embedding: project 4-d one-hot into embed_dim for additive conditioning
        self.rel_embed = nn.Sequential(
            nn.Linear(rel_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward projection to quantum angles
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_qubits),
        )
        self.norm2 = nn.LayerNorm(n_qubits)

    def forward(self, emb1, emb2, rels):
        """
        Args:
            emb1: (B, 512) face embedding for Person 1
            emb2: (B, 512) face embedding for Person 2
            rels: (B, 4) one-hot relation vector

        Returns:
            z1: (B, n_qubits) angle params for Person 1 in [-π, π]
            z2: (B, n_qubits) angle params for Person 2 in [-π, π]
        """
        # Add relation context to embeddings
        rel_bias = self.rel_embed(rels)  # (B, 512)
        e1 = emb1 + rel_bias  # (B, 512)
        e2 = emb2 + rel_bias  # (B, 512)

        # Reshape for attention: (B, 1, 512) — single-token sequence
        e1_seq = e1.unsqueeze(1)  # (B, 1, 512)
        e2_seq = e2.unsqueeze(1)  # (B, 1, 512)

        # Cross-attention: Person 1 attends to Person 2 and vice versa
        attn_out1, _ = self.cross_attn(e1_seq, e2_seq, e2_seq)  # (B, 1, 512)
        attn_out2, _ = self.cross_attn(e2_seq, e1_seq, e1_seq)  # (B, 1, 512)

        # Residual + LayerNorm
        h1 = self.norm1(e1 + attn_out1.squeeze(1))  # (B, 512)
        h2 = self.norm1(e2 + attn_out2.squeeze(1))  # (B, 512)

        # Project to quantum angle space
        z1 = self.norm2(self.projection(h1))  # (B, n_qubits)
        z2 = self.norm2(self.projection(h2))  # (B, n_qubits)

        # Scale to [-π, π]
        z1 = torch.tanh(z1) * math.pi
        z2 = torch.tanh(z2) * math.pi

        return z1, z2


# =============================================================================
# 3. LEGACY PROJECTION (Kept for ablation / backward compat)
# =============================================================================


class SimpleProjection(nn.Module):
    """
    Original 2-layer MLP projection: (512 + 4) → 128 → n_qubits.
    Kept for ablation comparison with the new CrossAttentionProjection.
    """

    def __init__(self, n_qubits=8):
        super().__init__()
        self.n_qubits = n_qubits
        self.projection = nn.Sequential(
            nn.Linear(512 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_qubits),
        )

    def forward(self, emb1, emb2, rels):
        x1 = torch.cat([emb1, rels], dim=1)
        x2 = torch.cat([emb2, rels], dim=1)
        z1 = torch.tanh(self.projection(x1)) * math.pi
        z2 = torch.tanh(self.projection(x2)) * math.pi
        return z1, z2


# =============================================================================
# 4. HYBRID KINSHIP CLASSIFIER (Rewritten)
# =============================================================================


class HybridKinshipClassifier(nn.Module):
    """
    Hybrid Classical-Quantum model for kinship verification using SWAP Test.

    Supports two encoding modes:
      - 'product':   Independent Ry encoding (classically simulable).
                     Uses analytical cos² formula for fast training.
      - 'entangled': Ry + CNOT chain + dual-learnable Rz encoding.
                     Uses differentiable statevector simulation with separate
                     register parameters so the fidelity can differ from the
                     product-state analytical formula.

    And two projection architectures:
      - 'cross_attention': CrossAttentionProjection (default, new)
      - 'simple':          SimpleProjection (legacy 2-layer MLP)
    """

    def __init__(
        self, n_qubits=8, encoding_mode="entangled", projection_type="cross_attention"
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoding_mode = encoding_mode
        self.projection_type = projection_type

        # Projection network
        if projection_type == "cross_attention":
            self.projection_net = CrossAttentionProjection(
                embed_dim=512, n_qubits=n_qubits, n_heads=8
            )
        else:
            self.projection_net = SimpleProjection(n_qubits=n_qubits)

        # Distinct phase parameters for each register in circuit-variant mode.
        if encoding_mode == "entangled":
            self.ent_params1 = nn.Parameter(
                torch.randn(n_qubits) * 0.1  # Small random init
            )
            self.ent_params2 = nn.Parameter(
                torch.randn(n_qubits) * 0.1  # Small random init
            )
        else:
            self.register_buffer("ent_params1", torch.zeros(n_qubits))
            self.register_buffer("ent_params2", torch.zeros(n_qubits))

    def forward(self, emb1, emb2, rels):
        """
        Differentiable forward pass.

        Args:
            emb1: (B, 512) face embeddings for Person 1
            emb2: (B, 512) face embeddings for Person 2
            rels: (B, 4) one-hot relation vectors

        Returns:
            fidelity: (B, 1) kinship probability in [0, 1]
        """
        # Project to quantum angle space
        z1, z2 = self.projection_net(emb1, emb2, rels)

        if self.encoding_mode == "entangled":
            # Differentiable shared-circuit fidelity with distinct register params.
            fidelity = differentiable_entangled_fidelity(
                z1, z2, self.ent_params1, self.ent_params2, self.n_qubits
            )
        else:
            # Fast analytical product-state fidelity
            fidelity = analytical_product_fidelity(z1, z2)

        return fidelity

    def forward_analytical(self, emb1, emb2, rels):
        """
        Product-state analytical fidelity (for comparison/ablation only).
        Always uses cos² formula regardless of encoding_mode.
        """
        z1, z2 = self.projection_net(emb1, emb2, rels)
        return analytical_product_fidelity(z1, z2)

    def forward_qiskit(self, emb1, emb2, rels, shots=1024):
        """
        Executes actual Qiskit SWAP test circuits on AerSimulator.
        Uses the appropriate circuit type based on encoding_mode.
        """
        self.eval()
        with torch.no_grad():
            z1, z2 = self.projection_net(emb1, emb2, rels)

        z1_np = z1.cpu().numpy()
        z2_np = z2.cpu().numpy()

        if self.encoding_mode == "entangled":
            ent1_np = self.ent_params1.detach().cpu().numpy()
            ent2_np = self.ent_params2.detach().cpu().numpy()
            fidelities = simulate_entangled_swap_test_batch(
                z1_np,
                z2_np,
                ent_params1=ent1_np,
                ent_params2=ent2_np,
                shots=shots,
            )
        else:
            fidelities = simulate_swap_test_batch(z1_np, z2_np, shots=shots)

        return torch.tensor(fidelities, dtype=torch.float32)

    def get_projected_angles(self, emb1, emb2, rels):
        """Returns the projected angle parameters (useful for analysis)."""
        with torch.no_grad():
            z1, z2 = self.projection_net(emb1, emb2, rels)
        return z1, z2

    def load_state_dict(self, state_dict, strict=True):
        """Support legacy checkpoints with single ent_params tensor."""
        state_dict = state_dict.copy()
        if "ent_params" in state_dict:
            ent = state_dict.pop("ent_params")
            state_dict["ent_params1"] = ent.clone()
            state_dict["ent_params2"] = ent.clone()
        return super().load_state_dict(state_dict, strict=strict)


class PairFusionKinshipClassifier(nn.Module):
    """
    Reliability-oriented supervised kinship classifier.

    This model keeps the strongest parts of the current pipeline -- frozen
    FaceNet embeddings, relation conditioning, and cross-pair interaction -- but
    does not force the final decision to be only a SWAP-test fidelity. It learns
    from symmetric pair features that are standard for verification tasks:
    absolute difference, elementwise product, cosine similarity, and relation
    context.
    """

    def __init__(self, embed_dim=512, hidden_dim=256, rel_dim=4, dropout=0.35):
        super().__init__()
        self.embed_dim = embed_dim
        self.rel_dim = rel_dim

        self.rel_embed = nn.Sequential(
            nn.Linear(rel_dim, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        self.shared_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Single-token cross-attention gives learned pair conditioning while
        # keeping the model compact and compatible with the existing idea.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        feature_dim = hidden_dim * 4 + rel_dim + 1
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

    def encode_pair(self, emb1, emb2, rels):
        rel_bias = self.rel_embed(rels)
        h1 = self.shared_proj(emb1 + rel_bias)
        h2 = self.shared_proj(emb2 + rel_bias)

        h1_seq = h1.unsqueeze(1)
        h2_seq = h2.unsqueeze(1)
        a1, _ = self.cross_attn(h1_seq, h2_seq, h2_seq)
        a2, _ = self.cross_attn(h2_seq, h1_seq, h1_seq)
        h1 = self.attn_norm(h1 + a1.squeeze(1))
        h2 = self.attn_norm(h2 + a2.squeeze(1))

        abs_diff = torch.abs(h1 - h2)
        prod = h1 * h2
        cosine = F.cosine_similarity(h1, h2, dim=1, eps=1e-8).unsqueeze(1)
        return torch.cat([h1, h2, abs_diff, prod, cosine, rels], dim=1)

    def forward_logits(self, emb1, emb2, rels):
        features = self.encode_pair(emb1, emb2, rels)
        return self.classifier(features)

    def forward(self, emb1, emb2, rels):
        return torch.sigmoid(self.forward_logits(emb1, emb2, rels))
