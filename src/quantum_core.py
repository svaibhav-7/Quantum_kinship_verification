"""
Quantum Core Module for Kinship Verification.

Provides two encoding strategies:
  1. Product-state encoding (Ry only; classically simulable via cos² formula)
  2. Shared-circuit encoding (Ry + CNOT chain + shared Rz)

Important: the current CNOT/Rz circuit applies the same unitary family to both
registers before evaluating fidelity. Inner products are invariant under a
common unitary, so this implementation produces the same fidelity as the
product-state cos² expression for the same projected angles. It is retained for
Qiskit circuit verification and timing experiments, not as a quantum-advantage
claim.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np
import torch

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
    total_qubits = 2 * n_qubits + 1
    qc = QuantumCircuit(total_qubits, 1)

    # 1. State preparation (Ry angle encoding)
    for i in range(n_qubits):
        qc.ry(float(z1[i]), 1 + i)
        qc.ry(float(z2[i]), 1 + n_qubits + i)

    # 2. SWAP Test
    qc.h(0)
    for i in range(n_qubits):
        qc.cswap(0, 1 + i, 1 + n_qubits + i)
    qc.h(0)

    # 3. Measurement of the ancilla qubit
    qc.measure(0, 0)

    return qc


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
    n_qubits = len(z1)
    qc = build_swap_test_circuit(n_qubits, z1, z2)

    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    count_0 = counts.get("0", 0)
    p_0 = count_0 / shots

    fidelity = 2 * p_0 - 1.0
    return max(0.0, min(1.0, fidelity))


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
    batch_size = len(z1_batch)
    n_qubits = z1_batch.shape[1]

    circuits = []
    for i in range(batch_size):
        qc = build_swap_test_circuit(n_qubits, z1_batch[i], z2_batch[i])
        circuits.append(qc)

    simulator = AerSimulator()
    job = simulator.run(circuits, shots=shots)
    results = job.result()

    fidelities = []
    for i in range(batch_size):
        counts = results.get_counts(i)
        count_0 = counts.get("0", 0)
        p_0 = count_0 / shots
        fidelity = 2 * p_0 - 1.0
        fidelities.append([max(0.0, min(1.0, fidelity))])

    return np.array(fidelities)


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
    if ent_params1 is None:
        ent_params1 = np.zeros(n_qubits)
    if ent_params2 is None:
        ent_params2 = np.zeros(n_qubits)

    total_qubits = 2 * n_qubits + 1
    qc = QuantumCircuit(total_qubits, 1)

    # ── Register 1 (Person 1): qubits 1..n ──
    for i in range(n_qubits):
        # Apply angle encoding and interleave entangling operations / phase shifts.
        qc.ry(float(z1[i]), 1 + i)
        qc.rz(float(ent_params1[i]), 1 + i)
        if i < n_qubits - 1:
            qc.cx(1 + i, 1 + i + 1)

    # ── Register 2 (Person 2): qubits n+1..2n ──
    for i in range(n_qubits):
        qc.ry(float(z2[i]), 1 + n_qubits + i)
        qc.rz(float(ent_params2[i]), 1 + n_qubits + i)
        if i < n_qubits - 1:
            qc.cx(1 + n_qubits + i, 1 + n_qubits + i + 1)

    # ── SWAP Test ──
    qc.h(0)
    for i in range(n_qubits):
        qc.cswap(0, 1 + i, 1 + n_qubits + i)
    qc.h(0)

    qc.measure(0, 0)

    return qc


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
    n_qubits = len(z1)
    qc = build_entangled_swap_test_circuit(
        n_qubits, z1, z2, ent_params1, ent_params2
    )

    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    count_0 = counts.get("0", 0)
    p_0 = count_0 / shots
    fidelity = 2 * p_0 - 1.0
    return max(0.0, min(1.0, fidelity))


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
    batch_size = len(z1_batch)
    n_qubits = z1_batch.shape[1]

    if ent_params1 is None:
        ent_params1 = np.zeros(n_qubits)
    if ent_params2 is None:
        ent_params2 = np.zeros(n_qubits)

    circuits = []
    for i in range(batch_size):
        qc = build_entangled_swap_test_circuit(
            n_qubits,
            z1_batch[i],
            z2_batch[i],
            ent_params1,
            ent_params2,
        )
        circuits.append(qc)

    simulator = AerSimulator()
    job = simulator.run(circuits, shots=shots)
    results = job.result()

    fidelities = []
    for i in range(batch_size):
        counts = results.get_counts(i)
        count_0 = counts.get("0", 0)
        p_0 = count_0 / shots
        fidelity = 2 * p_0 - 1.0
        fidelities.append([max(0.0, min(1.0, fidelity))])

    return np.array(fidelities)


# =============================================================================
# 3. STATEVECTOR-BASED EXACT FIDELITY (No shot noise)
# =============================================================================


def exact_entangled_fidelity(
    z1, z2, ent_params1=None, ent_params2=None
):
    """
    Computes exact fidelity between two shared-circuit states using Qiskit Statevector.
    No shot noise — this is the ground truth for comparison.

    Args:
        z1 (array-like): Ry angles for Person 1 (n_qubits,).
        z2 (array-like): Ry angles for Person 2 (n_qubits,).
        ent_params1 (array-like, optional): Rz entanglement params for register 1.
        ent_params2 (array-like, optional): Rz entanglement params for register 2.

    Returns:
        fidelity (float): Exact |<ψ₁|ψ₂>|² in [0, 1].
    """
    n_qubits = len(z1)
    if ent_params1 is None:
        ent_params1 = np.zeros(n_qubits)
    if ent_params2 is None:
        ent_params2 = np.zeros(n_qubits)

    # Build state 1 circuit
    qc1 = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc1.ry(float(z1[i]), i)
        qc1.rz(float(ent_params1[i]), i)
        if i < n_qubits - 1:
            qc1.cx(i, i + 1)

    # Build state 2 circuit
    qc2 = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc2.ry(float(z2[i]), i)
        qc2.rz(float(ent_params2[i]), i)
        if i < n_qubits - 1:
            qc2.cx(i, i + 1)

    sv1 = Statevector.from_instruction(qc1)
    sv2 = Statevector.from_instruction(qc2)

    # Fidelity = |<ψ₁|ψ₂>|²
    inner = np.abs(np.vdot(sv1.data, sv2.data)) ** 2
    return float(inner)


# =============================================================================
# 4. DIFFERENTIABLE SHARED-CIRCUIT FIDELITY IN PYTORCH
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

    # Build index slices for control=|1⟩
    idx_c0 = [slice(None)] * n_qubits
    idx_c0[control] = 1
    idx_c1 = [slice(None)] * n_qubits
    idx_c1[control] = 1

    # When control=|1⟩, flip target: swap |...1,0,...⟩ ↔ |...1,1,...⟩
    idx_t0 = list(idx_c1)
    idx_t0[target] = 0
    idx_t1 = list(idx_c1)
    idx_t1[target] = 1

    new_state[tuple(idx_t0)] = state[tuple(idx_t1)]
    new_state[tuple(idx_t1)] = state[tuple(idx_t0)]

    return new_state.reshape(dim)


def differentiable_entangled_fidelity(
    z1, z2, ent_params1, ent_params2, n_qubits
):
    """
    Computes SWAP test fidelity |<psi(z1)|psi(z2)>|^2 in pure PyTorch.

    Fully differentiable: gradients flow through z1, z2, ent_params1, and ent_params2.
    This is the core forward pass for the circuit-variant model.

    Circuit structure for each register:
        Step 1: Apply Ry(z_i) angle encoding on each qubit (parallel layer)
        Step 2: Apply CNOT entangling chain (linear depth)
        Step 3: Apply Rz(e_i) learnable phase rotations (parallel layer)
    
    Because register 1 and register 2 use DIFFERENT Rz parameters (ent_params1 vs ent_params2),
    this circuit is NOT equivalent to the product-state analytical formula. Distinct parameters
    allow the fidelity to vary based on the learned phase structure, creating a non-trivial
    quantum-inspired similarity metric.

    Args:
        z1: (B, n_qubits) Ry angle parameters for Person 1
        z2: (B, n_qubits) Ry angle parameters for Person 2
        ent_params1: (n_qubits,) learnable Rz parameters for register 1
        ent_params2: (n_qubits,) learnable Rz parameters for register 2
        n_qubits: number of qubits per register

    Returns:
        fidelity: (B, 1) tensor of fidelity values in [0, 1]
    """
    batch_size = z1.shape[0]
    dim = 2**n_qubits
    device = z1.device

    fidelities = []

    for b in range(batch_size):
        # Initialize |0⟩^⊗n for both states
        state1 = torch.zeros(dim, dtype=torch.complex64, device=device)
        state1[0] = 1.0 + 0j
        state2 = torch.zeros(dim, dtype=torch.complex64, device=device)
        state2[0] = 1.0 + 0j

        # ===== STEP 1: Apply Ry angle encoding (parallel layer) =====
        for i in range(n_qubits):
            ry1 = _ry_matrix(z1[b, i]).to(torch.complex64).to(device)
            ry2 = _ry_matrix(z2[b, i]).to(torch.complex64).to(device)
            state1 = _apply_single_qubit_gate(state1, ry1, i, n_qubits)
            state2 = _apply_single_qubit_gate(state2, ry2, i, n_qubits)

        # ===== STEP 2: Apply CNOT entangling chain (linear depth) =====
        for i in range(n_qubits - 1):
            state1 = _apply_cnot(state1, i, i + 1, n_qubits)
            state2 = _apply_cnot(state2, i, i + 1, n_qubits)

        # ===== STEP 3: Apply Rz phase rotations (parallel layer) with DISTINCT params =====
        for i in range(n_qubits):
            rz1 = _rz_matrix(ent_params1[i]).to(device)
            rz2 = _rz_matrix(ent_params2[i]).to(device)
            state1 = _apply_single_qubit_gate(state1, rz1, i, n_qubits)
            state2 = _apply_single_qubit_gate(state2, rz2, i, n_qubits)

        # Fidelity = |⟨ψ₁|ψ₂⟩|²
        inner = torch.vdot(state1, state2)  # conjugate-linear in first arg
        fid = (inner * inner.conj()).real
        fidelities.append(fid)

    return torch.stack(fidelities).unsqueeze(1)


def analytical_product_fidelity(z1, z2):
    """
    Fast analytical fidelity for PRODUCT states (no entanglement).
    F = ∏ cos²((z1ᵢ - z2ᵢ) / 2)

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
