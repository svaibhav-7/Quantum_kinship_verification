"""Fast, exact SWAP-test fidelity for the Ry -> CNOT-chain -> Rz ansatz.

The reference simulator materialises a 2^n statevector and permutes all n
dimensions once per qubit. That launches thousands of tiny kernels, so on an
RTX 5060 the quantum branch ran at 183 pairs/s while the classical path hit
240k pairs/s -- the GPU was idle, waiting on launch overhead.

A closed-form product over qubits does NOT work here: the shared CNOT chain
correlates the Rz phases, so the overlap is not factorizable. (Verified: with
Rz disabled a product form matches exactly; with distinct Rz it diverges.)
So this keeps the exact statevector maths and instead removes the overhead:

  * Ry layer built by reshaping to (B, left, 2, right) -- no permutes.
  * CNOT chain + Rz applied as one precomputed diagonal phase vector over all
    2^n basis states, so the entangling layer costs a single multiply.

Exactness against the reference simulator is pinned by
tests/test_fast_fidelity.py, including gradients.
"""

import torch


def _apply_ry_all(state, z, n_qubits):
    """Apply Ry(z_i) on every qubit. state: (B, 2^n) -> (B, 2^n)."""
    b = state.shape[0]
    for i in range(n_qubits):
        left, right = 2 ** i, 2 ** (n_qubits - i - 1)
        s = state.view(b, left, 2, right)
        c = torch.cos(z[:, i] / 2.0).view(b, 1, 1).to(state.dtype)
        sn = torch.sin(z[:, i] / 2.0).view(b, 1, 1).to(state.dtype)
        s0, s1 = s[:, :, 0, :], s[:, :, 1, :]
        state = torch.stack((c * s0 - sn * s1, sn * s0 + c * s1), dim=2).reshape(b, -1)
    return state


def _cnot_permutation(n_qubits, device):
    """Index map for the nearest-neighbour CNOT chain (control i -> target i+1)."""
    idx = torch.arange(2 ** n_qubits, device=device)
    out = idx.clone()
    for i in range(n_qubits - 1):
        cb = (n_qubits - 1) - i
        tb = (n_qubits - 1) - (i + 1)
        ctrl = (out >> cb) & 1
        out = out ^ (ctrl << tb)
    # Scatter: basis state j moves to out[j].
    perm = torch.empty_like(out)
    perm[out] = idx
    return perm


def _rz_phases(ent, n_qubits, device):
    """Diagonal phases e^{-i ent_i/2} / e^{+i ent_i/2} over all basis states."""
    idx = torch.arange(2 ** n_qubits, device=device)
    total = torch.zeros(2 ** n_qubits, device=device, dtype=ent.dtype)
    for i in range(n_qubits):
        bit = (idx >> ((n_qubits - 1) - i)) & 1
        total = total + torch.where(bit.bool(), ent[i] / 2.0, -ent[i] / 2.0)
    return torch.polar(torch.ones_like(total), total)


def fast_entangled_fidelity(z1, z2, ent_params1, ent_params2, n_qubits, ansatz_depth=1):
    """Exact fidelity |<psi(z1)|psi(z2)>|^2. Returns (B, 1)."""
    b = z1.shape[0]
    dim = 2 ** n_qubits
    device = z1.device

    s1 = torch.zeros(b, dim, dtype=torch.complex64, device=device)
    s1[:, 0] = 1.0
    s2 = s1.clone()

    s1 = _apply_ry_all(s1, z1[:, :n_qubits], n_qubits)
    s2 = _apply_ry_all(s2, z2[:, :n_qubits], n_qubits)

    perm = _cnot_permutation(n_qubits, device)
    for _ in range(ansatz_depth):
        s1 = s1[:, perm] * _rz_phases(ent_params1[:n_qubits], n_qubits, device)
        s2 = s2[:, perm] * _rz_phases(ent_params2[:n_qubits], n_qubits, device)

    overlap = (s1.conj() * s2).sum(dim=1)
    return (overlap.real ** 2 + overlap.imag ** 2).unsqueeze(1).clamp(0.0, 1.0)
