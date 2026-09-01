"""VQC classifiers and the capacity-matched control.

The control is what makes the experiment interpretable, so its parameter
matching is tested as carefully as the model itself: a control that is
accidentally smaller would make any quantum "win" meaningless.
"""
import os
import sys
import unittest

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_vqc import CapacityMatchedControl, VQCKinshipClassifier


def _batch(b=6, d=512):
    torch.manual_seed(0)
    return (torch.randn(b, d), torch.randn(b, d),
            torch.eye(4)[torch.randint(0, 4, (b,))])


class TestVQCClassifier(unittest.TestCase):
    def test_outputs_a_probability_per_pair(self):
        m = VQCKinshipClassifier(n_qubits_each=3, depth=2)
        out = m(*_batch())
        self.assertEqual(out.shape, (6, 1))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))

    def test_both_readouts_are_available(self):
        e1, e2, r = _batch()
        for readout, width in (("fidelity", 1), ("expectation", 6)):
            m = VQCKinshipClassifier(n_qubits_each=3, depth=2, readout=readout)
            self.assertEqual(m.measure(e1, e2, r).shape, (6, width))

    def test_rejects_an_unknown_readout(self):
        with self.assertRaises(ValueError):
            VQCKinshipClassifier(readout="teleportation")

    def test_predict_is_symmetric_in_argument_order(self):
        """The joint register permutes under a swap, so symmetry has to be
        imposed rather than assumed."""
        m = VQCKinshipClassifier(n_qubits_each=3, depth=2)
        m.eval()
        e1, e2, r = _batch()
        torch.testing.assert_close(m.predict(e1, e2, r), m.predict(e2, e1, r),
                                   rtol=1e-5, atol=1e-6)

    def test_raw_forward_is_not_symmetric(self):
        """Confirms the previous test is doing work: without the averaging the
        model genuinely depends on argument order."""
        m = VQCKinshipClassifier(n_qubits_each=3, depth=2)
        m.eval()
        e1, e2, r = _batch()
        with torch.no_grad():
            self.assertFalse(torch.allclose(m(e1, e2, r), m(e2, e1, r),
                                            atol=1e-6))

    def test_gradients_reach_the_circuit_parameters(self):
        m = VQCKinshipClassifier(n_qubits_each=3, depth=2)
        e1, e2, r = _batch()
        y = torch.randint(0, 2, (6, 1)).float()
        loss = torch.nn.functional.binary_cross_entropy(
            m(e1, e2, r).clamp(1e-6, 1 - 1e-6), y)
        loss.backward()
        self.assertGreater(float(m.vqc.theta.grad.abs().sum()), 0.0)

    def test_gradients_reach_the_encoder(self):
        m = VQCKinshipClassifier(n_qubits_each=3, depth=2)
        e1, e2, r = _batch()
        m(e1, e2, r).sum().backward()
        g = m.encoder.net[0].weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(float(g.abs().sum()), 0.0)

    def test_circuit_parameters_actually_move_when_trained(self):
        """A VQC that trains only its encoder would look like a null result."""
        m = VQCKinshipClassifier(n_qubits_each=3, depth=2)
        before = m.vqc.theta.detach().clone()
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        e1, e2, r = _batch()
        y = torch.randint(0, 2, (6, 1)).float()
        for _ in range(5):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy(
                m(e1, e2, r).clamp(1e-6, 1 - 1e-6), y).backward()
            opt.step()
        self.assertGreater(float((m.vqc.theta.detach() - before).abs().max()), 1e-4)


class TestCapacityMatchedControl(unittest.TestCase):
    def test_head_matches_the_circuit_parameter_count_closely(self):
        for n, d in ((3, 2), (4, 3), (4, 5), (2, 4)):
            vqc = VQCKinshipClassifier(n_qubits_each=n, depth=d)
            ctl = CapacityMatchedControl(n_qubits_each=n, depth=d)
            q, c = vqc.quantum_parameter_count(), ctl.head_parameter_count()
            self.assertLessEqual(abs(q - c) / q, 0.15,
                                 f"n={n} d={d}: quantum {q} vs control {c}")

    def test_control_is_never_much_smaller_than_the_circuit(self):
        """An undersized control would make a quantum win meaningless."""
        for n, d in ((3, 2), (4, 3), (4, 5)):
            vqc = VQCKinshipClassifier(n_qubits_each=n, depth=d)
            ctl = CapacityMatchedControl(n_qubits_each=n, depth=d)
            self.assertGreaterEqual(ctl.head_parameter_count(),
                                    0.85 * vqc.quantum_parameter_count())

    def test_encoder_is_identical_in_shape_to_the_vqc_encoder(self):
        """Only what happens after the angles should differ."""
        vqc = VQCKinshipClassifier(n_qubits_each=4, depth=3)
        ctl = CapacityMatchedControl(n_qubits_each=4, depth=3)
        a = {k: v.shape for k, v in vqc.encoder.state_dict().items()}
        b = {k: v.shape for k, v in ctl.encoder.state_dict().items()}
        self.assertEqual(a, b)

    def test_outputs_a_probability_and_is_symmetric(self):
        ctl = CapacityMatchedControl(n_qubits_each=3, depth=2)
        ctl.eval()
        e1, e2, r = _batch()
        out = ctl(e1, e2, r)
        self.assertEqual(out.shape, (6, 1))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))
        torch.testing.assert_close(ctl.predict(e1, e2, r),
                                   ctl.predict(e2, e1, r),
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
