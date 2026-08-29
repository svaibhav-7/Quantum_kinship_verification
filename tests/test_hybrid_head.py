"""The decision head must use pair features AND the quantum fidelity, so the
quantum contribution stays measurable as an ablation instead of being the
sole bottleneck."""
import os
import sys
import unittest
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_hybrid import QuantumAugmentedKinshipClassifier


class TestQuantumAugmentedHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.b = 16
        self.e1 = torch.randn(self.b, 512)
        self.e2 = torch.randn(self.b, 512)
        self.rel = torch.eye(4)[torch.randint(0, 4, (self.b,))]

    def test_outputs_probability_per_pair(self):
        m = QuantumAugmentedKinshipClassifier()
        out = m(self.e1, self.e2, self.rel)
        self.assertEqual(out.shape, (self.b, 1))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))

    def test_exposes_quantum_fidelity_for_ablation(self):
        m = QuantumAugmentedKinshipClassifier()
        fid = m.quantum_fidelity(self.e1, self.e2, self.rel)
        self.assertEqual(fid.shape, (self.b, 1))
        self.assertTrue(torch.all(fid >= 0) and torch.all(fid <= 1))

    def test_disabling_quantum_changes_the_prediction(self):
        """If use_quantum=False produced identical output, the quantum branch
        would be dead code and the ablation meaningless."""
        m = QuantumAugmentedKinshipClassifier()
        m.eval()
        with torch.no_grad():
            on = m(self.e1, self.e2, self.rel)
            m.use_quantum = False
            off = m(self.e1, self.e2, self.rel)
        self.assertFalse(torch.allclose(on, off))

    def test_is_symmetric_in_pair_order(self):
        """Kinship is symmetric; swapping the two faces must not change it."""
        m = QuantumAugmentedKinshipClassifier()
        m.eval()
        with torch.no_grad():
            ab = m(self.e1, self.e2, self.rel)
            ba = m(self.e2, self.e1, self.rel)
        torch.testing.assert_close(ab, ba, rtol=1e-4, atol=1e-4)

    def test_gradients_reach_the_quantum_parameters(self):
        """The old head starved these: grad norm ~0.04 vs ~20 on the MLP."""
        m = QuantumAugmentedKinshipClassifier()
        y = torch.randint(0, 2, (self.b, 1)).float()
        p = m(self.e1, self.e2, self.rel).clamp(1e-7, 1 - 1e-7)
        torch.nn.functional.binary_cross_entropy(p, y).backward()
        self.assertIsNotNone(m.ent_params1.grad)
        self.assertGreater(m.ent_params1.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
