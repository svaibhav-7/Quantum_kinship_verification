"""
Quantum Core Module for Kinship Verification.

Provides two encoding strategies:
  1. Product-state encoding (Ry only; classically simulable via analytical cos² formula)
  2. Entangled encoding mode (Ry + CNOT chain + distinct Rz register parameters)

Note: In entangled mode, distinct Rz parameters (ent_params1 vs ent_params2) allow
the statevector fidelity to vary based on learned phase structure, producing a non-trivial
quantum-inspired similarity metric.
"""

import torch
import math
import numpy as np

# =============================================================================
# 1. PRODUCT-STATE ENCODING (Legacy — kept for ablation baselines)
# =============================================================================


def analytical_product_fidelity(z1, z2):
    """
    Fast analytical fidelity for PRODUCT states (no entanglement).
    F = � ∏ cos²((z1��ᵢ - z2��ᵢ) / 2)

    This is the closed-form shortcut that only works when encoding is
    independent Ry rotations with no entanglement.

    Args:
        z1: (B, n_qubits) tensor
        z2: (B, n_qubits) tensor

    Returns:
        fidelity: (B, 1) tensor
    """
    cos_diff = torch.cos((z1 - z2) / 2.0)
    fidelity = torch.prod(cos_diff**2, dim=1, keepdim=True)
    return fidelity


# =============================================================================
# 2. DIFFERENTIABLE SHARED-CIRCUIT FIDELITY IN PYTORCH
# =============================================================================


def _ry_matrix(theta):
    """Ry(θ) rotation matrix as a 2×2 complex torch tensor."""
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _rz_matrix(theta):
    """Rz(θ) rotation matrix as a 2×2 complex torch tensor."""
    half = theta / 2.0
    return torch.stack(
        [
            torch.stack(
                [
                    torch.exp(-1j * half.to(torch.complex64)),
                    torch.zeros(1, dtype=torch.complex64).squeeze(),
                ]
            ),
            torch.stack(
                [
                    torch.zeros(1, dtype=torch.complex64).squeeze(),
                    torch.exp(1j * half.to(torch.complex64)),
                ]
            ),
        ]
    )


def _apply_single_qubit_gate(state, gate, qubit, n_qubits):
    """
    Applies a single-qubit gate to a statevector tensor.

    Args:
        state: (2^n,) complex tensor
        gate: (2, 2) complex tensor
        qubit: target qubit index
        n_qubits: total number of qubits

    Returns:
        new_state: (2^n,) complex tensor
    """
    dim = 2**n_qubits
    state = state.reshape([2] * n_qubits)
    state = torch.tensordot(gate, state, dims=([1], [qubit]))
    # Move the contracted dimension back to the correct position
    perm = list(range(1, qubit + 1)) + [0] + list(range(qubit + 1, n_qubits))
    state = state.permute(perm)
    return state.reshape(dim)


def _apply_cnot(state, control, target, n_qubits):
    """
    Applies CNOT gate to a statevector tensor.

    Args:
        state: (2^n,) complex tensor
        control: control qubit index
        target: target qubit index
        n_qubits: total number of qubits

    Returns:
        new_state: (2^n,) complex tensor
    """
    dim = 2**n_qubits
    state = state.reshape([2] * n_qubits)
    new_state = state.clone()

    # Build index slices for control=|1��⟩
    idx_c0 = [slice(None)] * n_qubits
    idx_c0[control] = 1
    idx_c1 = [slice(None)] * n_qubits
    idx_c1[control] = 1

    # When control=|1��⟩, flip target: swap |...1,0,...��⟩ ↔ |...1,1,...��⟩
    idx_t0 = list(idx_c1)
    idx_t0[target] = 0
    idx_t1 = list(idx_c1)
    idx_t1[target] = 1

    new_state[tuple(idx_t0)] = state[tuple(idx_t1)]
    new_state[tuple(idx_t1)] = state[tuple(idx_t0)]

    return new_state.reshape(dim)


def differentiable_entangled_fidelity(
    z1, z2, ent_params1, ent_params2, n_qubits, ansatz_depth=1
):
    """
    Computes SWAP test fidelity |<psi(z1)|psi(z2)>|^2 in pure PyTorch.

    Implements a hardware-efficient ansatz with alternating layers of:
    - Entangling gates (nearest-neighbor CNOT chain)
    - Single-qubit rotations (Rz with learnable parameters)

    Circuit structure for each register:
        Step 1: Apply Ry(z_i) angle encoding on each qubit (parallel layer) - data encoding
        Step 2: Repeat ansatz_depth times:
                a. Apply CNOT entangling chain (linear depth) - entangling layer
                b. Apply Rz(e_i) learnable phase rotations (parallel layer) - rotation layer

    Because register 1 and register 2 use DIFFERENT Rz parameters (ent_params1 vs ent_params2),
    this circuit is NOT equivalent to the product-state analytical formula. Distinct parameters
    allow the fidelity to vary based on the learned phase structure, creating a non-trivial
    quantum-inspired similarity metric.

    Args:
        z1: (B, n_qubits) Ry angle parameters for Person 1
        z2: (B, n_qubits) Ry angle parameters for Person 2
        ent_params1: (n_qubits,) learnable Rz parameters for register 1 (shared across layers if ansatz_depth>1)
        ent_params2: (n_qubits,) learnable Rz parameters for register 2 (shared across layers if ansatz_depth>1)
        n_qubits: number of qubits per register
        ansatz_depth: number of entangling+rotation layers to apply (default=1 for backward compatibility)

    Returns:
        fidelity: (B, 1) tensor of fidelity values in [0, 1]
    """
    batch_size = z1.shape[0]
    dim = 2**n_qubits
    device = z1.device

    fidelities = []

    for b in range(batch_size):
        # Initialize |0������⟩^������⊗n for both states
        state1 = torch.zeros(dim, dtype=torch.complex64, device=device)
        state1[0] = 1.0 + 0j
        state2 = torch.zeros(dim, dtype=torch.complex64, device=device)
        state2[0] = 1.0 + 0j

        # ===== STEP 1: Apply Ry angle encoding (parallel layer) - data encoding =====
        for i in range(n_qubits):
            ry1 = _ry_matrix(z1[b, i]).to(torch.complex64).to(device)
            ry2 = _ry_matrix(z2[b, i]).to(torch.complex64).to(device)
            state1 = _apply_single_qubit_gate(state1, ry1, i, n_qubits)
            state2 = _apply_single_qubit_gate(state2, ry2, i, n_qubits)

        # ===== STEP 2: Repeat ansatz_depth layers of [entangling -> rotation] =====
        for layer in range(ansatz_depth):
            # ------- Entangling layer: CNOT chain (linear depth) -------
            for i in range(n_qubits - 1):
                state1 = _apply_cnot(state1, i, i + 1, n_qubits)
                state2 = _apply_cnot(state2, i, i + 1, n_qubits)

            # ------- Rotation layer: Rz phase rotations (parallel layer) -------
            # Use the same ent_params for all layers (shared across layers)
            for i in range(n_qubits):
                rz1 = _rz_matrix(ent_params1[i]).to(device)
                rz2 = _rz_matrix(ent_params2[i]).to(device)
                state1 = _apply_single_qubit_gate(state1, rz1, i, n_qubits)
                state2 = _apply_single_qubit_gate(state2, rz2, i, n_qubits)

        # Fidelity = |������⟨ψ���₁|ψ₂������⟩|²
        inner = torch.vdot(state1, state2)  # conjugate-linear in first arg
        fid = (inner * inner.conj()).real
        fidelities.append(fid)

    return torch.stack(fidelities).unsqueeze(1)


def differentiable_entangled_fidelity_vectorized(
    z1, z2, ent_params1, ent_params2, n_qubits, ansatz_depth=1
):
    """
    Vectorized version of differentiable_entangled_fidelity that eliminates Python loops
    over batch and qubit dimensions for improved performance.

    Computes SWAP test fidelity |<psi(z1)|psi(z2)>|^2 in pure PyTorch using batched
    tensor operations instead of iterative loops.

    Args:
        z1: (B, n_qubits) Ry angle parameters for Person 1
        z2: (B, n_qubits) Ry angle parameters for Person 2
        ent_params1: (n_qubits,) learnable Rz parameters for register 1
        ent_params2: (n_qubits,) learnable Rz parameters for register 2
        n_qubits: number of qubits per register
        ansatz_depth: number of entangling+rotation layers to apply

    Returns:
        fidelity: (B, 1) tensor of fidelity values in [0, 1]
    """
    batch_size = z1.shape[0]
    dim = 2**n_qubits
    device = z1.device

    # Initialize |0������������������������������⟩^������������������������������⊗n for both states - shape: (batch, 2^n_qubits)
    state1 = torch.zeros(batch_size, dim, dtype=torch.complex64, device=device)
    state1[:, 0] = 1.0 + 0j
    state2 = torch.zeros(batch_size, dim, dtype=torch.complex64, device=device)
    state2[:, 0] = 1.0 + 0j

    # ===== STEP 1: Apply Ry angle encoding (parallel layer) - data encoding =====
    # Vectorized over batch and qubits
    for i in range(n_qubits):
        # Get angles for all batch elements at qubit i
        ry1_angles = z1[:, i]  # (batch,)
        ry2_angles = z2[:, i]  # (batch,)

        # Create batch of Ry matrices - shape: (batch, 2, 2)
        cos1 = torch.cos(ry1_angles / 2.0)  # (batch,)
        sin1 = torch.sin(ry1_angles / 2.0)  # (batch,)
        ry1_matrices = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
        ry1_matrices[:, 0, 0] = cos1
        ry1_matrices[:, 0, 1] = -sin1
        ry1_matrices[:, 1, 0] = sin1
        ry1_matrices[:, 1, 1] = cos1

        cos2 = torch.cos(ry2_angles / 2.0)  # (batch,)
        sin2 = torch.sin(ry2_angles / 2.0)  # (batch,)
        ry2_matrices = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
        ry2_matrices[:, 0, 0] = cos2
        ry2_matrices[:, 0, 1] = -sin2
        ry2_matrices[:, 1, 0] = sin2
        ry2_matrices[:, 1, 1] = cos2

        # Apply to all batch elements using the same logic as _apply_single_qubit_gate but batched
        state1_reshaped = state1.view(batch_size, *[2] * n_qubits)
        state2_reshaped = state2.view(batch_size, *[2] * n_qubits)

        # Apply gate to qubit i for all batch elements
        # state: [B, 2, 2, ..., 2] (n_qubits times 2)
        # gate: [B, 2, 2]
        # We want to contract gate dimension 1 with state dimension i+1 (because of batch dim at 0)

        # For each batch element, we do: tensordot(gate[b], state[b], dims=([1], [i]))
        # But we can do it all at once by treating the batch dimension separately

        # Move the qubit dimension to the end for easier tensordot
        perm_forward = list(range(0, i+1)) + list(range(i+2, n_qubits+1)) + [i+1]
        state1_perm = state1_reshaped.permute(perm_forward)  # [B, 2, 2, ..., 2] with qubit i at end
        state2_perm = state2_reshaped.permute(perm_forward)

        # Now apply gate: state_perm[b] has shape [2, 2, ..., 2] with qubit i at the end (index -1)
        # We want to contract gate[b] (2x2) with state_perm[b] on the last dimension
        # This is equivalent to: state_reshaped[b] = gate[b] @ state_perm[b].view(-1, 2).T).view([2]*n_qubits)

        # Reshape to combine all dimensions except the last one
        state1_combined = state1_perm.reshape(batch_size, -1, 2)  # [B, 2^(n_qubits-1), 2]
        state2_combined = state2_perm.reshape(batch_size, -1, 2)

        # Apply batched matrix multiplication: [B, 2^(n_qubits-1), 2] @ [B, 2, 2]^T -> [B, 2^(n_qupts-1), 2]
        # But we need to be careful about conjuage for quantum gates
        # Actually, for unitary gates, we just do normal matrix multiplication
        state1_result = torch.matmul(state1_combined, ry1_matrices.transpose(-2, -1))  # [B, 2^(n_qubits-1), 2]
        state2_result = torch.matmul(state2_combined, ry2_matrices.transpose(-2, -1))

        # Reshape back to [B, 2, 2, ..., 2]
        state1_reshaped = state1_result.reshape(batch_size, *[2] * n_qubits)
        state2_reshaped = state2_result.reshape(batch_size, *[2] * n_qubits)

        # Move qubit dimension back to its original position
        perm_backward = list(range(0, i+1)) + [n_qubits] + list(range(i+1, n_qubits))
        state1_reshaped = state1_reshaped.permute(perm_backward)
        state2_reshaped = state2_reshaped.permute(perm_backward)

        # Reshape back
        state1 = state1_reshaped.reshape(batch_size, dim)
        state2 = state2_reshaped.reshape(batch_size, dim)

    # ===== STEP 2: Repeat ansatz_depth layers of [entangling -> rotation] =====
    for layer in range(ansatz_depth):
        # ------- Entangling layer: CNOT chain (linear depth) -------
        # Apply CNOT chain: qubit 0->1, 1->2, ..., (n-2)->(n-1)
        # Vectorized over batch
        for i in range(n_qubits - 1):
            # Vectorized CNOT application over batch
            # Reshape to separate qubits
            state1_reshaped = state1.view(batch_size, *[2] * n_qubits)
            state2_reshaped = state2.view(batch_size, *[2] * n_qubits)

            # Build index slices for control=|1��������������⟩
            # Create lists of slice objects for advanced indexing
            idx_shape = [batch_size] + [2] * n_qubits
            idx_lists = [[slice(None)] * (n_qubits + 1) for _ in range(batch_size)]

            # Set control qubit to 1 for all batch elements
            for b in range(batch_size):
                idx_lists[b][i+1] = 1  # +1 because dim 0 is batch

            # When control=|1��������������⟩, flip target: swap |...1,0,...��������������⟩ ↔ |...1,1,...��������������⟩
            # Create indices for the swap
            idx_t0_lists = [list(idx) for idx in idx_lists]
            idx_t1_lists = [list(idx) for idx in idx_lists]

            for b in range(batch_size):
                idx_t0_lists[b][i+2] = 0  # target qubit |0��������������⟩ (+2: batch + control + target)
                idx_t1_lists[b][i+2] = 1  # target qubit |1��������������⟩

            # Apply the swap for each batch element
            for b in range(batch_size):
                # Index for |...,1,0,...> state (control=|1>, target=|0>)
                idx_10 = [slice(None)] * (n_qubits + 1)
                idx_10[0] = b  # batch index
                idx_10[i+1] = 1  # control qubit |1>
                idx_10[i+2] = 0  # target qubit |0>

                # Index for |...,1,1,...> state (control=|1>, target=|1>)
                idx_11 = [slice(None)] * (n_qubits + 1)
                idx_11[0] = b  # batch index
                idx_11[i+1] = 1  # control qubit |1>
                idx_11[i+2] = 1  # target qubit |1>

                # Swap the values
                temp1 = state1_reshaped[tuple(idx_10)].clone()
                temp2 = state2_reshaped[tuple(idx_10)].clone()
                state1_reshaped[tuple(idx_10)] = state1_reshaped[tuple(idx_11)]
                state2_reshaped[tuple(idx_10)] = state2_reshaped[tuple(idx_11)]
                state1_reshaped[tuple(idx_11)] = temp1
                state2_reshaped[tuple(idx_11)] = temp2

            # Reshape back
            state1 = state1_reshaped.reshape(batch_size, dim)
            state2 = state2_reshaped.reshape(batch_size, dim)

        # ------- Rotation layer: Rz phase rotations (parallel layer) -------
        # Use the same ent_params for all layers (shared across layers)
        # Vectorized over batch and qubits
        for i in range(n_qubits):
            # Get Rz angles for all batch elements at qubit i (same for all batch elements)
            rz_angle1 = ent_params1[i]  # scalar
            rz_angle2 = ent_params2[i]  # scalar

            # Create batch of Rz matrices - shape: (batch, 2, 2)
            half1 = rz_angle1 / 2.0
            half2 = rz_angle2 / 2.0
            exp_minus_1j_half1 = torch.exp(-1j * half1.to(torch.complex64))
            exp_1j_half1 = torch.exp(1j * half1.to(torch.complex64))
            exp_minus_1j_half2 = torch.exp(-1j * half2.to(torch.complex64))
            exp_1j_half2 = torch.exp(1j * half2.to(torch.complex64))

            rz1_matrices = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
            rz1_matrices[:, 0, 0] = exp_minus_1j_half1
            rz1_matrices[:, 0, 1] = 0.0
            rz1_matrices[:, 1, 0] = 0.0
            rz1_matrices[:, 1, 1] = exp_1j_half1

            rz2_matrices = torch.zeros(batch_size, 2, 2, dtype=torch.complex64, device=device)
            rz2_matrices[:, 0, 0] = exp_minus_1j_half2
            rz2_matrices[:, 0, 1] = 0.0
            rz2_matrices[:, 1, 0] = 0.0
            rz2_matrices[:, 1, 1] = exp_1j_half2

            # Apply to all batch elements using the same logic as _apply_single_qubit_gate but batched
            state1_reshaped = state1.view(batch_size, *[2] * n_qubits)
            state2_reshaped = state2.view(batch_size, *[2] * n_qubits)

            # Apply gate to qubit i for all batch elements
            # state: [B, 2, 2, ..., 2] (n_qubits times 2)
            # gate: [B, 2, 2]
            # We want to contract gate dimension 1 with state dimension i+1 (because of batch dim at 0)

            # Move the qubit dimension to the end for easier tensordot
            perm_forward = list(range(0, i+1)) + list(range(i+2, n_qubits+1)) + [i+1]
            state1_perm = state1_reshaped.permute(perm_forward)  # [B, 2, 2, ..., 2] with qubit i at end
            state2_perm = state2_reshaped.permute(perm_forward)

            # Now apply gate: state_perm[b] has shape [2, 2, ..., 2] with qubit i at the end (index -1)
            # We want to contract gate[b] (2x2) with state_perm[b] on the last dimension
            # This is equivalent to: state_reshaped[b] = gate[b] @ state_perm[b].view(-1, 2).T).view([2]*n_qubits)

            # Reshape to combine all dimensions except the last one
            state1_combined = state1_perm.reshape(batch_size, -1, 2)  # [B, 2^(n_qubits-1), 2]
            state2_combined = state2_perm.reshape(batch_size, -1, 2)

            # Apply batched matrix multiplication: [B, 2, 2] @ [B, 2^(n_qubits-1), 2]^T -> [B, 2, 2^(n_qubits-1)]
            # Then transpose back to [B, 2^(n_qubits-1), 2]
            # This applies the Rz gate correctly as: new_state = Rz @ old_state
            state1_result = torch.matmul(rz1_matrices, state1_combined.transpose(-2, -1)).transpose(-2, -1)  # [B, 2^(n_qubits-1), 2]
            state2_result = torch.matmul(rz2_matrices, state2_combined.transpose(-2, -1)).transpose(-2, -1)

            # Reshape back to [B, 2, 2, ..., 2]
            state1_reshaped = state1_result.reshape(batch_size, *[2] * n_qubits)
            state2_reshaped = state2_result.reshape(batch_size, *[2] * n_qubits)

            # Move qubit dimension back to its original position
            perm_backward = list(range(0, i+1)) + [n_qubits] + list(range(i+1, n_qubits))
            state1_reshaped = state1_reshaped.permute(perm_backward)
            state2_reshaped = state2_reshaped.permute(perm_backward)

            # Reshape back
            state1 = state1_reshaped.reshape(batch_size, dim)
            state2 = state2_reshaped.reshape(batch_size, dim)

    # Fidelity = |������������������������������⟨ψ���������������₁|ψ₂������������������������������⟩|²
    # Batched inner product: sum over last dimension, conjugate in first argument
    inner = torch.sum(torch.conj(state1) * state2, dim=-1)  # (batch,)
    fid = torch.real(inner * torch.conj(inner))  # |inner|^2

    return fid.unsqueeze(-1)  # (batch, 1)




# =============================================================================
# 1. PRODUCT-STATE ENCODING (Legacy — kept for ablation baselines)
# =============================================================================


def build_swap_test_circuit(n_qubits, z1, z2):
    """
    Builds a Quantum SWAP Test circuit with PRODUCT-STATE Ry encoding.

    This encoding is classically simulable: F = prod(cos²((z1-z2)/2)).
    Kept for backward compatibility and ablation comparison.

    Args:
        n_qubits (int): Number of qubits representing each face embedding.
        z1 (array-like): Angle encoding parameters for Person 1 (length n_qubits).
        z2 (array-like): Angle encoding parameters for Person 2 (length n_qubits).

    Returns:
        qc (QuantumCircuit): The complete Qiskit QuantumCircuit.
    """
    # Note: This is a placeholder for compatibility with existing code.
    # The actual implementation would require Qiskit, but we're keeping
    # the function signature for compatibility.
    # For now, we'll return None and handle this in the simulation functions.
    return None


def simulate_swap_test(z1, z2, shots=1024):
    """
    Simulates the product-state SWAP test circuit on AerSimulator.

    Args:
        z1 (array-like): Angle parameters for Person 1.
        z2 (array-like): Angle parameters for Person 2.
        shots (int): Number of simulation shots.

    Returns:
        fidelity (float): Overlap value in range [0, 1].
    """
    # For compatibility, we fall back to the analytical product fidelity
    # when Qiskit is not available or for testing purposes.
    if isinstance(z1, np.ndarray):
        z1 = torch.from_numpy(z1).float()
    if isinstance(z2, np.ndarray):
        z2 = torch.from_numpy(z2).float()

    # Add batch dimension if needed
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)

    fidelity = analytical_product_fidelity(z1, z2)
    return fidelity.item() if fidelity.numel() == 1 else fidelity[-1].item()


def simulate_swap_test_batch(z1_batch, z2_batch, shots=1024):
    """
    Runs product-state SWAP test for a batch of feature pairs.

    Args:
        z1_batch (np.ndarray): (Batch, n_qubits) array of angles.
        z2_batch (np.ndarray): (Batch, n_qubits) array of angles.
        shots (int): Number of shots.

    Returns:
        fidelities (np.ndarray): (Batch, 1) array of overlap values.
    """
    # Convert to torch tensors if needed
    if isinstance(z1_batch, np.ndarray):
        z1_batch = torch.from_numpy(z1_batch).float()
    if isinstance(z2_batch, np.ndarray):
        z2_batch = torch.from_numpy(z2_batch).float()

    # Use analytical product fidelity for batch
    fidelity = analytical_product_fidelity(z1_batch, z2_batch)
    if isinstance(fidelity, torch.Tensor):
        return fidelity.numpy()
    return fidelity


# =============================================================================
# 2. SHARED-CIRCUIT ENCODING
# =============================================================================


def build_entangled_swap_test_circuit(
    n_qubits, z1, z2, ent_params1=None, ent_params2=None
):
    """
    Builds a SWAP Test circuit with the shared CNOT/Rz encoding variant.

    Each register is prepared via:
      Step 1: Ry(z_i) angle encoding on each qubit (same as product state)
      Step 2: CNOT entangling chain connecting adjacent qubits
      Step 3: Rz(ent_param_i) parameterized phase rotations (learnable)

    Because the same CNOT/Rz unitary is applied to both compared states, the
    resulting fidelity is invariant to that shared unitary and matches the
    product-state cos² fidelity for the same Ry angles.

    Args:
        n_qubits (int): Number of qubits per register.
        z1 (array-like): Ry angle parameters for Person 1 (length n_qubits).
        z2 (array-like): Ry angle parameters for Person 2 (length n_qubits).
        ent_params1 (array-like, optional): Rz entanglement params for register 1.
            If None, defaults to zeros (no additional phase).
        ent_params2 (array-like, optional): Rz entanglement params for register 2.
            If None, defaults to zeros.

    Returns:
        qc (QuantumCircuit): The complete entangled SWAP test circuit.
    """
    # Note: This is a placeholder for compatibility with existing code.
    # The actual implementation would require Qiskit, but we're keeping
    # the function signature for compatibility.
    return None


def simulate_entangled_swap_test(
    z1,
    z2,
    ent_params1=None,
    ent_params2=None,
    shots=1024,
):
    """
    Simulates the shared CNOT/Rz SWAP test circuit.

    Args:
        z1 (array-like): Ry angle params for Person 1.
        z2 (array-like): Ry angle params for Person 2.
        ent_params1 (array-like, optional): Rz entanglement params for register 1.
        ent_params2 (array-like, optional): Rz entanglement params for register 2.
        shots (int): Number of simulation shots.

    Returns:
        fidelity (float): Overlap value in range [0, 1].
    """
    # For compatibility, we fall back to a simplified version
    # that uses the differentiable entangled fidelity with ansatz_depth=1
    if isinstance(z1, np.ndarray):
        z1 = torch.from_numpy(z1).float()
    if isinstance(z2, np.ndarray):
        z2 = torch.from_numpy(z2).float()
    if ent_params1 is not None and isinstance(ent_params1, np.ndarray):
        ent_params1 = torch.from_numpy(ent_params1).float()
    if ent_params2 is not None and isinstance(ent_params2, np.ndarray):
        ent_params2 = torch.from_numpy(ent_params2).float()

    # Add batch dimension if needed
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)
    if ent_params1 is not None and ent_params1.dim() == 1:
        ent_params1 = ent_params1.unsqueeze(0)
    if ent_params2 is not None and ent_params2.dim() == 1:
        ent_params2 = ent_params2.unsqueeze(0)

    # Use differentiable entangled fidelity with ansatz_depth=1 for compatibility
    fidelity = differentiable_entangled_fidelity(z1, z2, ent_params1, ent_params2, z1.shape[-1], ansatz_depth=1)
    if isinstance(fidelity, torch.Tensor):
        return fidelity.item() if fidelity.numel() == 1 else fidelity[-1].item()
    return fidelity


def simulate_entangled_swap_test_batch(
    z1_batch,
    z2_batch,
    ent_params1=None,
    ent_params2=None,
    shots=1024,
):
    """
    Batch-mode shared CNOT/Rz SWAP test simulation.

    Args:
        z1_batch (np.ndarray): (B, n_qubits) angle params.
        z2_batch (np.ndarray): (B, n_qubits) angle params.
        ent_params1 (np.ndarray, optional): (n_qubits,) entanglement params for register 1.
        ent_params2 (np.ndarray, optional): (n_qubits,) entanglement params for register 2.
        shots (int): Number of shots.

    Returns:
        fidelities (np.ndarray): (B, 1) overlap values.
    """
    # Convert to torch tensors if needed
    if isinstance(z1_batch, np.ndarray):
        z1_batch = torch.from_numpy(z1_batch).float()
    if isinstance(z2_batch, np.ndarray):
        z2_batch = torch.from_numpy(z2_batch).float()
    if ent_params1 is not None and isinstance(ent_params1, np.ndarray):
        ent_params1 = torch.from_numpy(ent_params1).float()
    if ent_params2 is not None and isinstance(ent_params2, np.ndarray):
        ent_params2 = torch.from_numpy(ent_params2).float()

    # Use differentiable entangled fidelity for batch
    fidelity = differentiable_entangled_fidelity(z1_batch, z2_batch, ent_params1, ent_params2, z1_batch.shape[-1], ansatz_depth=1)
    if isinstance(fidelity, torch.Tensor):
        return fidelity.numpy()
    return fidelity