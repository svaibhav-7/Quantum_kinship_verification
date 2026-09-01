"""Joint-register variational quantum circuit for kinship verification.

Implements the architecture in
`docs/superpowers/specs/2026-09-01-variational-quantum-classifier-design.md`:
both people's encodings occupy one `2n`-qubit register, a variational `U(theta)`
acts on the whole register, and the decision is read from a measurement.

This differs from `quantum_fast.py` in the way that matters. That module encodes
each person separately, applies a fixed Rz layer to each, and compares the two
states -- so the two people never interact until the final overlap. Here the
CNOT ring includes bonds crossing between the halves, so `U(theta)` can form
correlations *between* the two people. Whether that is worth anything is the
experimental question; the point is that it is expressible here and was not
before.

Everything is exact statevector simulation in PyTorch, differentiable
end-to-end. Correctness is pinned against an independently constructed
reference in `tests/test_quantum_vqc.py`.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_angles(z, n_qubits):
    """Angle-encode `z` as a product state: Ry(z_i) on |0> for each qubit.

    Builds the product incrementally rather than via an explicit Kronecker
    chain, which keeps it batched and differentiable. Returns (B, 2**n_qubits).
    """
    b = z.shape[0]
    state = torch.ones(b, 1, dtype=torch.complex64, device=z.device)
    for i in range(n_qubits):
        a = z[:, i] / 2.0
        q = torch.stack([torch.cos(a), torch.sin(a)], dim=1).to(torch.complex64)
        # outer product with the running state, then flatten
        state = (state.unsqueeze(2) * q.unsqueeze(1)).reshape(b, -1)
    return state


def amplitude_encode(v, n_qubits):
    """Encode a real vector directly as statevector amplitudes.

    Angle encoding puts `n` numbers into `n` qubits -- one Ry angle each --
    which forces a 512-d embedding through an `n`-dimensional bottleneck before
    the circuit sees anything. Amplitude encoding puts `2**n` numbers into the
    same `n` qubits, so at n=9 the whole FaceNet embedding is carried with no
    compression at all.

    The state is the L2-normalised input, zero-padded to `2**n`. Cosine
    similarity between two inputs is therefore preserved exactly as the
    inner product between their encoded states.
    """
    dim = 2 ** n_qubits
    v = v[:, :dim] if v.shape[1] >= dim else torch.nn.functional.pad(
        v, (0, dim - v.shape[1]))
    v = v / (torch.linalg.vector_norm(v, dim=1, keepdim=True) + 1e-12)
    return v.to(torch.complex64)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _apply_1q(state, mats, qubit, n_qubits):
    """Apply a batch of 2x2 gates to `qubit` of a (B, 2**n) statevector."""
    b = state.shape[0]
    left, right = 2 ** qubit, 2 ** (n_qubits - qubit - 1)
    s = state.reshape(b, left, 2, right)
    s0, s1 = s[:, :, 0, :], s[:, :, 1, :]
    m = mats  # (B, 2, 2)
    out0 = m[:, 0, 0].view(b, 1, 1) * s0 + m[:, 0, 1].view(b, 1, 1) * s1
    out1 = m[:, 1, 0].view(b, 1, 1) * s0 + m[:, 1, 1].view(b, 1, 1) * s1
    return torch.stack([out0, out1], dim=2).reshape(b, -1)


def _ry_rz(theta_y, theta_z, batch, device):
    """Rz(theta_z) Ry(theta_y) as one 2x2, broadcast over the batch."""
    cy, sy = torch.cos(theta_y / 2), torch.sin(theta_y / 2)
    ry = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])])
    ry = ry.to(torch.complex64)

    half = theta_z / 2
    e_m = torch.complex(torch.cos(-half), torch.sin(-half))
    e_p = torch.complex(torch.cos(half), torch.sin(half))
    zero = torch.zeros((), dtype=torch.complex64, device=device)
    rz = torch.stack([torch.stack([e_m, zero]), torch.stack([zero, e_p])])

    m = rz @ ry
    return m.unsqueeze(0).expand(batch, 2, 2)


def _cnot_indices(n_qubits, control, target, device):
    """Permutation mapping basis states under CNOT(control -> target)."""
    idx = torch.arange(2 ** n_qubits, device=device)
    cbit = (idx >> (n_qubits - 1 - control)) & 1
    flip = cbit << (n_qubits - 1 - target)
    return idx ^ flip


# ---------------------------------------------------------------------------
# The circuit
# ---------------------------------------------------------------------------

class JointRegisterVQC(nn.Module):
    """Variational circuit over a joint 2n-qubit register.

    Layout: person 1 on qubits 0..n-1, person 2 on qubits n..2n-1.

    Each layer applies Ry(theta) Rz(theta) to every qubit, then a CNOT ring
    0->1->...->2n-1->0. The ring is what makes this a joint model: the bonds
    (n-1 -> n) and (2n-1 -> 0) cross between the two people's halves. Remove
    them and the circuit factorises into the independent-register case that
    earlier formulations already tested.

    Parameters: 2 * 2n * depth.
    """

    def __init__(self, n_qubits_each=4, depth=3):
        super().__init__()
        self.n_each = n_qubits_each
        self.n_total = 2 * n_qubits_each
        self.depth = depth
        # (depth, n_total, 2) -- Ry and Rz angle per qubit per layer.
        # Small init keeps the circuit near identity at the start, which
        # avoids the barren-plateau regime a large random init can produce.
        self.theta = nn.Parameter(torch.randn(depth, self.n_total, 2) * 0.1)

    def joint_state(self, z1, z2):
        """Encode both people into one register and apply U(theta)."""
        z = torch.cat([z1, z2], dim=1)
        state = encode_angles(z, self.n_total)
        b, dev = state.shape[0], state.device

        for d in range(self.depth):
            for q in range(self.n_total):
                m = _ry_rz(self.theta[d, q, 0], self.theta[d, q, 1], b, dev)
                state = _apply_1q(state, m, q, self.n_total)
            for q in range(self.n_total):
                perm = _cnot_indices(self.n_total, q,
                                     (q + 1) % self.n_total, dev)
                state = state[:, perm]
        return state

    def fidelity(self, z1, z2):
        """|<0...0| U(theta) |psi>|^2 -- the single-scalar readout.

        This is the literal reading of the diagram's "Fidelity" branch, and
        also the readout an earlier diagnosis on this project found to be
        lossy: compressing the decision to one number cost roughly 16 ROC-AUC
        points there. Reported alongside `expectations` so the two can be
        compared rather than assumed equivalent.
        """
        s = self.joint_state(z1, z2)
        amp = s[:, 0]
        return (amp.real ** 2 + amp.imag ** 2).unsqueeze(1).clamp(0.0, 1.0)

    def expectations(self, z1, z2):
        """<Z> per qubit -- the vector readout. Returns (B, 2n)."""
        s = self.joint_state(z1, z2)
        probs = (s.real ** 2 + s.imag ** 2)          # (B, 2**n_total)
        b = probs.shape[0]

        outs = []
        for q in range(self.n_total):
            left, right = 2 ** q, 2 ** (self.n_total - q - 1)
            p = probs.reshape(b, left, 2, right)
            outs.append(p[:, :, 0, :].sum(dim=(1, 2)) - p[:, :, 1, :].sum(dim=(1, 2)))
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# Independent reference, used only by tests
# ---------------------------------------------------------------------------

def reference_statevector(z1, z2, theta, n_each, depth):
    """Same circuit built from explicit Kronecker products.

    Deliberately naive and independent of the batched implementation above.
    An earlier optimisation on this project was mathematically invalid and was
    caught only by a comparison of this kind, so the redundancy is the point.
    """
    n = 2 * n_each
    dim = 2 ** n
    z = torch.cat([z1, z2], dim=1)
    out = []

    theta = theta.detach()   # reference is a check, not a gradient path
    z = z.detach()

    for b in range(z.shape[0]):
        state = torch.zeros(dim, dtype=torch.complex64)
        state[0] = 1.0
        for i in range(n):
            a = z[b, i] / 2
            g = torch.tensor([[torch.cos(a), -torch.sin(a)],
                              [torch.sin(a), torch.cos(a)]], dtype=torch.complex64)
            full = torch.eye(1, dtype=torch.complex64)
            for j in range(n):
                full = torch.kron(full, g if j == i else torch.eye(2, dtype=torch.complex64))
            state = full @ state

        for d in range(depth):
            for q in range(n):
                ty, tz = theta[d, q, 0], theta[d, q, 1]
                cy, sy = torch.cos(ty / 2), torch.sin(ty / 2)
                ry = torch.tensor([[cy, -sy], [sy, cy]], dtype=torch.complex64)
                h = tz / 2
                rz = torch.tensor(
                    [[torch.complex(torch.cos(-h), torch.sin(-h)), 0],
                     [0, torch.complex(torch.cos(h), torch.sin(h))]],
                    dtype=torch.complex64)
                g = rz @ ry
                full = torch.eye(1, dtype=torch.complex64)
                for j in range(n):
                    full = torch.kron(full, g if j == q else torch.eye(2, dtype=torch.complex64))
                state = full @ state

            for q in range(n):
                ctrl, tgt = q, (q + 1) % n
                cn = torch.zeros(dim, dim, dtype=torch.complex64)
                for k in range(dim):
                    cbit = (k >> (n - 1 - ctrl)) & 1
                    kk = k ^ (cbit << (n - 1 - tgt))
                    cn[kk, k] = 1.0
                state = cn @ state
        out.append(state)
    return torch.stack(out)
