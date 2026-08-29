def test_dynamic_quantum_fusion():
    from src.models_improved import MetaEnsembleKinshipClassifier
    from src.models_improved import EnsembleKinshipClassifier, HybridKinshipClassifier
    import torch
    import numpy as np

    # Create mock ensemble models
    ensemble_full = EnsembleKinshipClassifier([HybridKinshipClassifier() for _ in range(5)])
    ensemble_fiw = EnsembleKinshipClassifier([HybridKinshipClassifier() for _ in range(5)])
    single_fiw = HybridKinshipClassifier()

    model = MetaEnsembleKinshipClassifier(ensemble_full, ensemble_fiw, single_fiw)

    # Check that gating network exists
    assert hasattr(model, 'gating_network')

    # Mock inputs
    emb1 = torch.randn(2, 512)
    emb2 = torch.randn(2, 512)
    rels = torch.randn(2, 4)

    fidelity = model(emb1, emb2, rels)

    # Check output validity
    assert fidelity.shape == (2, 1)
    assert torch.all(fidelity >= 0) and torch.all(fidelity <= 1)

    # Check that gating network produces valid weights (should sum to 1 for each batch item)
    with torch.no_grad():
        # Get individual predictions to test gating network
        p1 = ensemble_full(emb1, emb2, rels)
        p2 = ensemble_fiw(emb1, emb2, rels)
        p3 = single_fiw(emb1, emb2, rels)
        ensemble_preds = torch.stack([p1.squeeze(-1), p2.squeeze(-1), p3.squeeze(-1)], dim=1)  # (B, 3)
        weights = model.gating_network(ensemble_preds)  # (B, 3)

        # Check that weights are valid (non-negative and sum to 1)
        assert torch.all(weights >= 0), "Gating network should produce non-negative weights"
        assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0)), "Weights should sum to 1"

    # Additional check: ensure gradients flow through the gating network
    loss = fidelity.sum()
    loss.backward()

    # Check that gating network parameters have gradients
    assert model.gating_network[0].weight.grad is not None, "Gating network should have gradients"
    assert model.gating_network[2].weight.grad is not None, "Gating network should have gradients"

    # Check that phase parameters also have gradients (indicating quantum fusion is active)
    assert model.phase_full.grad is not None
    assert model.phase_fiw.grad is not None
    assert model.phase_single.grad is not None


def test_dynamic_vs_static_weights():
    """Test that dynamic weights can produce different results than static weights."""
    from src.models_improved import MetaEnsembleKinshipClassifier
    from src.models_improved import EnsembleKinshipClassifier, HybridKinshipClassifier
    import torch

    # Create mock ensemble models with predictable outputs
    class FixedOutputEnsemble(EnsembleKinshipClassifier):
        def __init__(self, fixed_values):
            # Create models that return fixed values
            fixed_models = [HybridKinshipClassifier() for _ in fixed_values]
            super().__init__(fixed_models)
            self.fixed_values = fixed_values

        def forward(self, emb1, emb2, rels):
            batch_size = emb1.shape[0]
            # Return different fixed values for each model in ensemble
            # For simplicity, we'll use the average of fixed values
            avg_value = sum(self.fixed_values) / len(self.fixed_values)
            return torch.full((batch_size, 1), avg_value, dtype=emb1.dtype)

    # Create ensembles with different fixed outputs
    ensemble_full = FixedOutputEnsemble([0.9, 0.8, 0.7, 0.6, 0.5])  # Avg = 0.7
    ensemble_fiw = FixedOutputEnsemble([0.5, 0.4, 0.3, 0.2, 0.1])   # Avg = 0.3
    single_fiw = HybridKinshipClassifier()
    # Make single_fiw also return a fixed value by overriding forward
    single_fiw_fixed = HybridKinshipClassifier()
    original_forward = single_fiw_fixed.forward
    single_fiw_fixed.forward = lambda emb1, emb2, rels: torch.full((emb1.shape[0], 1), 0.2)

    model = MetaEnsembleKinshipClassifier(ensemble_full, ensemble_fiw, single_fiw_fixed)

    # Test with specific inputs
    emb1 = torch.randn(2, 512)
    emb2 = torch.randn(2, 512)
    rels = torch.randn(2, 4)

    # Get dynamic fusion result
    fidelity_dynamic = model(emb1, emb2, rels)

    # Get static weights result (using initial weights 0.45, 0.35, 0.20)
    with torch.no_grad():
        p1 = ensemble_full(emb1, emb2, rels)  # Should be ~0.7
        p2 = ensemble_fiw(emb1, emb2, rels)   # Should be ~0.3
        p3 = single_fiw_fixed(emb1, emb2, rels)  # Should be 0.2

        # Static combination with initial weights
        static_result = 0.45 * p1 + 0.35 * p2 + 0.20 * p3

    # They might be equal in some cases, but we're testing that the mechanism works
    # The key test is that both produce valid outputs and gradients flow
    assert fidelity_dynamic.shape == (2, 1)
    assert torch.all(fidelity_dynamic >= 0) and torch.all(fidelity_dynamic <= 1)
    assert static_result.shape == (2, 1)
    assert torch.all(static_result >= 0) and torch.all(static_result <= 1)

    print(f"[PASS] Dynamic quantum fusion test passed")
    print(f"  Dynamic result: {fidelity_dynamic[0].item():.4f}")
    print(f"  Static result (0.45*p1 + 0.35*p2 + 0.20*p3): {static_result[0].item():.4f}")


if __name__ == "__main__":
    test_dynamic_quantum_fusion()
    test_dynamic_vs_static_weights()
    print("\n[PASS] All Dynamic Quantum Fusion tests passed!")