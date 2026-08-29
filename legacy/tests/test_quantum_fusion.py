def test_quantum_state_fusion():
    from src.models_improved import MetaEnsembleKinshipClassifier
    from src.models_improved import EnsembleKinshipClassifier, HybridKinshipClassifier
    import torch

    # Create mock ensemble models
    ensemble_full = EnsembleKinshipClassifier([HybridKinshipClassifier() for _ in range(5)])
    ensemble_fiw = EnsembleKinshipClassifier([HybridKinshipClassifier() for _ in range(5)])
    single_fiw = HybridKinshipClassifier()

    model = MetaEnsembleKinshipClassifier(ensemble_full, ensemble_fiw, single_fiw)

    # Mock inputs
    emb1 = torch.randn(2, 512)
    emb2 = torch.randn(2, 512)
    rels = torch.randn(2, 4)

    fidelity = model(emb1, emb2, rels)

    # Check output validity
    assert fidelity.shape == (2, 1)
    assert torch.all(fidelity >= 0) and torch.all(fidelity <= 1)

    # Check that quantum fusion produces different results than classical averaging
    # (at least for some inputs, due to quantum interference)
    with torch.no_grad():
        # Get individual predictions
        p1 = ensemble_full(emb1, emb2, rels)
        p2 = ensemble_fiw(emb1, emb2, rels)
        p3 = single_fiw(emb1, emb2, rels)

        # Classical weighted average (using same weights as initialization)
        w1, w2, w3 = 0.45, 0.35, 0.20
        classical_avg = w1 * p1 + w2 * p2 + w3 * p3

        # Quantum fusion should differ from classical averaging due to interference
        # (though they might be equal for some specific inputs, they should not always be equal)
        # We'll check that the model is doing something non-trivial
        assert not torch.allclose(fidelity, classical_avg, atol=1e-3), \
            "Quantum fusion should produce different results from classical averaging"

    # Additional check: ensure gradients flow through the quantum fusion
    loss = fidelity.sum()
    loss.backward()

    # Check that phase parameters have gradients (indicating quantum fusion is active)
    assert model.phase_full.grad is not None
    assert model.phase_fiw.grad is not None
    assert model.phase_single.grad is not None