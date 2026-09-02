"""Joint-register variational quantum circuit.

A VQC fails silently in two ways: it can implement the wrong unitary and still
produce plausible numbers, or its parameters can receive no gradient and train
to the encoder's performance while looking like a null result. Both are guarded
here.

The reference comparison is not a formality. An earlier optimisation on this
project was mathematically invalid -- a closed-form product that ignored how a
shared CNOT chain correlates phases -- and only a comparison against an
explicitly constructed reference caught it.
"""
import os
import sys
import unittest

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum_vqc import (JointRegisterVQC, encode_angles,
                             reference_statevector)


class TestAngleEncoding(unittest.TestCase):
    def test_produces_a_normalised_state(self):
        z = torch.randn(4, 8) * np.pi
        s = encode_angles(z, n_qubits=8)
        norms = torch.linalg.vector_norm(s, dim=1)
        torch.testing.assert_close(norms, torch.ones(4), rtol=1e-5, atol=1e-6)

    def test_state_dimension_is_two_to_the_n(self):
        s = encode_angles(torch.zeros(2, 6), n_qubits=6)
        self.assertEqual(s.shape, (2, 64))

    def test_zero_angles_give_the_all_zeros_basis_state(self):
        s = encode_angles(torch.zeros(1, 4), n_qubits=4)
        self.assertAlmostEqual(abs(s[0, 0].item()), 1.0, places=6)
        self.assertAlmostEqual(float(s[0, 1:].abs().sum()), 0.0, places=6)

    def test_matches_an_explicit_kronecker_construction(self):
        """The independent reference: build the product state by hand."""
        torch.manual_seed(0)
        z = torch.randn(3, 4)
        got = encode_angles(z, n_qubits=4)
        for b in range(3):
            want = torch.tensor([1.0 + 0j])
            for i in range(4):
                a = z[b, i] / 2
                q = torch.tensor([torch.cos(a), torch.sin(a)], dtype=torch.complex64)
                want = torch.kron(want, q)
            torch.testing.assert_close(got[b], want, rtol=1e-4, atol=1e-5)


class TestJointRegisterCircuit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n = 3           # per person
        self.total = 2 * self.n
        self.vqc = JointRegisterVQC(n_qubits_each=self.n, depth=2)
        self.z1 = torch.randn(5, self.n)
        self.z2 = torch.randn(5, self.n)

    def test_joint_state_spans_both_registers(self):
        s = self.vqc.joint_state(self.z1, self.z2)
        self.assertEqual(s.shape, (5, 2 ** self.total))

    def test_state_stays_normalised_after_the_circuit(self):
        s = self.vqc.joint_state(self.z1, self.z2)
        norms = torch.linalg.vector_norm(s, dim=1)
        torch.testing.assert_close(norms, torch.ones(5), rtol=1e-4, atol=1e-5)

    def test_depth_zero_reduces_to_plain_angle_encoding(self):
        plain = JointRegisterVQC(n_qubits_each=self.n, depth=0)
        got = plain.joint_state(self.z1, self.z2)
        want = encode_angles(torch.cat([self.z1, self.z2], dim=1), self.total)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    def test_matches_reference_simulator(self):
        """Independent construction from explicit gate matrices."""
        for depth in (1, 2):
            vqc = JointRegisterVQC(n_qubits_each=2, depth=depth)
            torch.manual_seed(7)
            with torch.no_grad():
                vqc.theta.copy_(torch.randn_like(vqc.theta))
            z1, z2 = torch.randn(2, 2), torch.randn(2, 2)
            got = vqc.joint_state(z1, z2)
            want = reference_statevector(z1, z2, vqc.theta, n_each=2, depth=depth)
            torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4,
                                       msg=f"depth={depth}")

    def test_the_entangling_ring_actually_entangles_the_two_halves(self):
        """If the circuit factorised across the two people it would reduce to
        the already-tested independent-register case."""
        vqc = JointRegisterVQC(n_qubits_each=2, depth=1)
        with torch.no_grad():
            vqc.theta.fill_(0.7)
        z1 = torch.tensor([[0.9, 0.3]])
        z2 = torch.tensor([[0.2, 1.1]])
        s = vqc.joint_state(z1, z2)[0].reshape(4, 4)   # split the two halves
        # A product state across the cut has Schmidt rank 1.
        rank = int((torch.linalg.svdvals(s) > 1e-5).sum())
        self.assertGreater(rank, 1, "circuit factorises; the ring is not coupling")


class TestReadouts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.vqc = JointRegisterVQC(n_qubits_each=3, depth=2)
        self.z1, self.z2 = torch.randn(6, 3), torch.randn(6, 3)

    def test_fidelity_is_a_probability(self):
        f = self.vqc.fidelity(self.z1, self.z2)
        self.assertEqual(f.shape, (6, 1))
        self.assertTrue(torch.all(f >= -1e-6) and torch.all(f <= 1 + 1e-6))

    def test_expectations_are_one_per_qubit_and_in_range(self):
        e = self.vqc.expectations(self.z1, self.z2)
        self.assertEqual(e.shape, (6, 6))          # 2n qubits
        self.assertTrue(torch.all(e >= -1 - 1e-5) and torch.all(e <= 1 + 1e-5))

    def test_expectation_of_a_basis_state_is_all_plus_one(self):
        """|00...0> has <Z> = +1 on every qubit."""
        vqc = JointRegisterVQC(n_qubits_each=2, depth=0)
        e = vqc.expectations(torch.zeros(1, 2), torch.zeros(1, 2))
        torch.testing.assert_close(e, torch.ones(1, 4), rtol=1e-4, atol=1e-5)

    def test_expectations_carry_more_information_than_the_scalar(self):
        """The point of running both readouts: the vector should not be a
        deterministic function of the scalar."""
        f = self.vqc.fidelity(self.z1, self.z2).squeeze(1)
        e = self.vqc.expectations(self.z1, self.z2)
        # If every column were perfectly rank-correlated with f, the vector
        # would add nothing.
        cors = []
        for j in range(e.shape[1]):
            c = np.corrcoef(f.detach().numpy(), e[:, j].detach().numpy())[0, 1]
            cors.append(abs(c) if np.isfinite(c) else 0.0)
        self.assertLess(min(cors), 0.999)


class TestGradients(unittest.TestCase):
    """A VQC whose parameters do not move is the most likely silent failure."""

    def test_theta_receives_gradient_from_the_fidelity_readout(self):
        vqc = JointRegisterVQC(n_qubits_each=3, depth=2)
        out = vqc.fidelity(torch.randn(4, 3), torch.randn(4, 3)).sum()
        out.backward()
        self.assertIsNotNone(vqc.theta.grad)
        self.assertGreater(float(vqc.theta.grad.abs().sum()), 0.0)

    def test_theta_receives_gradient_from_the_expectation_readout(self):
        vqc = JointRegisterVQC(n_qubits_each=3, depth=2)
        out = vqc.expectations(torch.randn(4, 3), torch.randn(4, 3)).sum()
        out.backward()
        self.assertIsNotNone(vqc.theta.grad)
        self.assertGreater(float(vqc.theta.grad.abs().sum()), 0.0)

    def test_gradient_reaches_the_input_angles(self):
        """The encoder upstream must be trainable through the circuit."""
        vqc = JointRegisterVQC(n_qubits_each=3, depth=1)
        z1 = torch.randn(4, 3, requires_grad=True)
        z2 = torch.randn(4, 3, requires_grad=True)
        vqc.fidelity(z1, z2).sum().backward()
        self.assertGreater(float(z1.grad.abs().sum()), 0.0)
        self.assertGreater(float(z2.grad.abs().sum()), 0.0)

    def test_parameter_count_matches_the_declared_formula(self):
        for n, d in ((3, 2), (4, 3), (2, 1)):
            vqc = JointRegisterVQC(n_qubits_each=n, depth=d)
            self.assertEqual(vqc.theta.numel(), 2 * (2 * n) * d)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_gives_the_same_circuit(self):
        torch.manual_seed(3)
        a = JointRegisterVQC(n_qubits_each=3, depth=2)
        torch.manual_seed(3)
        b = JointRegisterVQC(n_qubits_each=3, depth=2)
        torch.testing.assert_close(a.theta, b.theta)

    def test_repeated_evaluation_is_identical(self):
        vqc = JointRegisterVQC(n_qubits_each=3, depth=2)
        z1, z2 = torch.randn(4, 3), torch.randn(4, 3)
        torch.testing.assert_close(vqc.fidelity(z1, z2), vqc.fidelity(z1, z2))


if __name__ == "__main__":
    unittest.main()


class TestAmplitudeEncoding(unittest.TestCase):
    """Angle encoding puts n numbers into n qubits; amplitude encoding puts
    2^n. At n=9 that is the whole 512-d FaceNet embedding with no compression,
    which removes the bottleneck that angle encoding imposes upstream of the
    circuit."""

    def test_encodes_a_full_vector_without_compression(self):
        from src.quantum_vqc import amplitude_encode

        v = torch.randn(4, 512)
        s = amplitude_encode(v, n_qubits=9)
        self.assertEqual(s.shape, (4, 512))

    def test_state_is_normalised(self):
        from src.quantum_vqc import amplitude_encode

        s = amplitude_encode(torch.randn(6, 64), n_qubits=6)
        torch.testing.assert_close(torch.linalg.vector_norm(s, dim=1),
                                   torch.ones(6), rtol=1e-5, atol=1e-6)

    def test_preserves_direction_up_to_scale(self):
        """The encoded state is the input vector normalised, so cosine
        similarity between inputs survives encoding exactly."""
        from src.quantum_vqc import amplitude_encode

        a, b = torch.randn(1, 64), torch.randn(1, 64)
        ca = torch.nn.functional.cosine_similarity(a, b).item()
        sa, sb = amplitude_encode(a, 6), amplitude_encode(b, 6)
        cb = torch.real((sa.conj() * sb).sum()).item()
        self.assertAlmostEqual(ca, cb, places=5)

    def test_pads_when_the_vector_is_shorter_than_the_register(self):
        from src.quantum_vqc import amplitude_encode

        s = amplitude_encode(torch.randn(2, 100), n_qubits=7)  # 128 slots
        self.assertEqual(s.shape, (2, 128))
        torch.testing.assert_close(torch.linalg.vector_norm(s, dim=1),
                                   torch.ones(2), rtol=1e-5, atol=1e-6)

    def test_gradients_flow_through_the_encoding(self):
        from src.quantum_vqc import amplitude_encode

        v = torch.randn(3, 64, requires_grad=True)
        amplitude_encode(v, 6).abs().sum().backward()
        self.assertGreater(float(v.grad.abs().sum()), 0.0)
