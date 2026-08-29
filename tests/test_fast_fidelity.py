"""A fast fidelity path is only acceptable if it reproduces the reference
statevector simulator exactly. These tests pin that equivalence."""
import os, sys, unittest, torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum_core import differentiable_entangled_fidelity_vectorized
from src.quantum_fast import fast_entangled_fidelity


class TestFastMatchesReference(unittest.TestCase):
    def _case(self, batch, nq, seed):
        g = torch.Generator().manual_seed(seed)
        z1 = (torch.rand(batch, nq, generator=g) * 2 - 1) * 3.14159
        z2 = (torch.rand(batch, nq, generator=g) * 2 - 1) * 3.14159
        e1 = torch.randn(nq, generator=g) * 0.5
        e2 = torch.randn(nq, generator=g) * 0.5
        return z1, z2, e1, e2

    def test_matches_reference_across_shapes(self):
        for batch, nq, seed in [(8, 4, 0), (16, 8, 1), (5, 6, 2), (1, 8, 3)]:
            z1, z2, e1, e2 = self._case(batch, nq, seed)
            ref = differentiable_entangled_fidelity_vectorized(z1, z2, e1, e2, nq).view(-1)
            fast = fast_entangled_fidelity(z1, z2, e1, e2, nq).view(-1)
            torch.testing.assert_close(fast, ref, rtol=1e-4, atol=1e-5,
                                       msg=f"mismatch batch={batch} nq={nq}")

    def test_identical_states_give_unit_fidelity(self):
        z1, _, e1, e2 = self._case(8, 8, 4)
        f = fast_entangled_fidelity(z1, z1, e1, e1, 8).view(-1)
        torch.testing.assert_close(f, torch.ones_like(f), rtol=1e-4, atol=1e-5)

    def test_output_is_a_valid_probability(self):
        z1, z2, e1, e2 = self._case(32, 8, 5)
        f = fast_entangled_fidelity(z1, z2, e1, e2, 8)
        self.assertTrue(torch.all(f >= -1e-6) and torch.all(f <= 1 + 1e-6))

    def test_gradients_match_reference(self):
        z1, z2, e1, e2 = self._case(6, 6, 6)
        z1r, z2r = z1.clone().requires_grad_(), z2.clone().requires_grad_()
        differentiable_entangled_fidelity_vectorized(z1r, z2r, e1, e2, 6).sum().backward()
        z1f, z2f = z1.clone().requires_grad_(), z2.clone().requires_grad_()
        fast_entangled_fidelity(z1f, z2f, e1, e2, 6).sum().backward()
        torch.testing.assert_close(z1f.grad, z1r.grad, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
