import unittest
import torch
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models_improved import HybridKinshipClassifier


class TestPhysicsRegularization(unittest.TestCase):
    def test_forward_pass_and_physics_reg(self):
        model = HybridKinshipClassifier(
            n_qubits=8,
            encoding_mode="entangled",
            projection_type="quantum_inspired_attention"
        )

        batch_size = 4
        emb1 = torch.randn(batch_size, 512)
        emb2 = torch.randn(batch_size, 512)
        rels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

        output = model(emb1, emb2, rels)
        self.assertEqual(output.shape, (batch_size, 1))

        physics_loss = model.physics_regularization()
        self.assertIsInstance(physics_loss, torch.Tensor)


if __name__ == "__main__":
    unittest.main()