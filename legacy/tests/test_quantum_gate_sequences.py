def test_learnable_quantum_gate_sequences():
    from src.models_improved import QuantumInspiredCrossAttention
    import torch

    batch_size, embed_dim, n_qubits, n_heads = 4, 512, 8, 8
    model = QuantumInspiredCrossAttention(embed_dim=embed_dim, n_qubits=n_qubits, n_heads=n_heads)

    emb1 = torch.randn(batch_size, embed_dim)
    emb2 = torch.randn(batch_size, embed_dim)
    rels = torch.randn(batch_size, 4)

    z1, z2 = model(emb1, emb2, rels)

    # Check that quantum gate sequences parameters exist
    assert hasattr(model, 'quantum_gate_sequences')
    assert len(model.quantum_gate_sequences) == n_heads
    assert model.quantum_gate_sequences[0].shape == (n_qubits, 3)

    # Check that output is still valid
    assert z1.shape == (batch_size, n_qubits)
    assert z2.shape == (batch_size, n_qubits)