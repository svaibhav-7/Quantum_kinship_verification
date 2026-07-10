import torch
import sys
sys.path.insert(0, 'D:/SasiVaibhav/klu/3rd year/projects/Quantum_kinship')

from src.models_improved import HybridKinshipClassifier
import numpy as np

# Test the physics regularization fix
print("Testing physics regularization fix...")

try:
    # Create model with entangled encoding and quantum-inspired attention
    model = HybridKinshipClassifier(
        n_qubits=8,
        encoding_mode="entangled",
        projection_type="quantum_inspired_attention"
    )

    # Create dummy inputs
    batch_size = 4
    emb1 = torch.randn(batch_size, 512)
    emb2 = torch.randn(batch_size, 512)
    rels = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

    # Test forward pass
    print("Testing forward pass...")
    output = model(emb1, emb2, rels)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Test physics regularization
    print("Testing physics regularization...")
    physics_loss = model.physics_regularization()
    print(f"Physics loss: {physics_loss.item():.6f}")

    print("SUCCESS: All tests passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()