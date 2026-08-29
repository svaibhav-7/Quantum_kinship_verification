def test_hierarchical_quantum_interference():
    from src.models_improved import QuantumInspiredCrossAttention
    import torch

    batch_size, embed_dim, n_qubits = 4, 512, 8
    model = QuantumInspiredCrossAttention(embed_dim=embed_dim, n_qubits=n_qubits)

    emb1 = torch.randn(batch_size, embed_dim)
    emb2 = torch.randn(batch_size, embed_dim)
    rels = torch.randn(batch_size, 4)

    z1, z2 = model(emb1, emb2, rels)

    # Check that output has correct shape and range
    assert z1.shape == (batch_size, n_qubits)
    assert z2.shape == (batch_size, n_qubits)
    assert torch.all(z1 >= -torch.pi) and torch.all(z1 <= torch.pi)
    assert torch.all(z2 >= -torch.pi) and torch.all(z2 <= torch.pi)

    # Basic sanity check: output should not be all zeros
    assert not torch.all(z1 == 0)
    assert not torch.all(z2 == 0)

    # Check that phase matrices are learnable (have gradients)
    emb1.requires_grad_(True)
    emb2.requires_grad_(True)
    rels.requires_grad_(True)

    z1, z2 = model(emb1, emb2, rels)
    loss = z1.sum() + z2.sum()
    loss.backward()

    # Check that phase matrices have gradients (they are being used)
    assert model.phase_matrix1.grad is not None
    assert model.phase_matrix2.grad is not None
    assert torch.any(model.phase_matrix1.grad != 0)
    assert torch.any(model.phase_matrix2.grad != 0)