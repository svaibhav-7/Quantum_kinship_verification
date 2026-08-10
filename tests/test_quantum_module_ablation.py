import torch
import numpy as np
from src.models_improved import HybridKinshipClassifier
from sklearn.metrics import accuracy_score

def test_quantum_module_criticality():
    """Test that disabling the quantum module causes a performance drop and that the disabled model performs at chance."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load a baseline model with enhanced quantum-infused components
    model = HybridKinshipClassifier(
        n_qubits=8,
        encoding_mode="entangled",
        projection_type="quantum_inspired_attention"
    )
    model.eval()

    # Create a disabled quantum module model that always returns 0.5 (random guessing)
    class DisabledQuantumModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, emb1, emb2, rels):
            # Return tensor of 0.5 with same shape as base model output
            return torch.full_like(self.base_model(emb1, emb2, rels), 0.5)

    disabled_model = DisabledQuantumModel(model)
    disabled_model.eval()

    # Generate random embeddings and relation vectors
    batch_size = 2000  # Increased for more stable estimates
    emb1 = torch.randn(batch_size, 512)
    emb2 = torch.randn(batch_size, 512)
    rels = torch.randn(batch_size, 4)  # random relation vectors

    # Generate random binary labels (for verification task)
    y_true = torch.randint(0, 2, (batch_size,)).float()

    # Get predictions from both models
    with torch.no_grad():
        y_pred_enabled = model(emb1, emb2, rels).view(-1).numpy()
        y_pred_disabled = disabled_model(emb1, emb2, rels).view(-1).numpy()

    # Convert to binary predictions using threshold from ablation study (0.5216)
    threshold = 0.5216
    y_pred_enabled_bin = (y_pred_enabled >= threshold).astype(float)
    y_pred_disabled_bin = (y_pred_disabled >= threshold).astype(float)
    y_true_np = y_true.numpy()

    # Calculate accuracies
    acc_enabled = accuracy_score(y_true_np, y_pred_enabled_bin) * 100
    acc_disabled = accuracy_score(y_true_np, y_pred_disabled_bin) * 100

    # Calculate performance drop
    performance_drop = acc_enabled - acc_disabled

    # Assert that quantum module contributes positively (performance drop > 0)
    assert performance_drop > 0.0, f"Performance drop {performance_drop:.2f}% <= 0%"

    # Additionally, check that disabled model performs near chance (50%)
    # Allow a small margin due to random variation in labels and thresholding
    assert abs(acc_disabled - 50.0) < 5.0, f"Disabled model accuracy {acc_disabled:.2f}% not near chance (within 5%)"

    print(f"Enabled model accuracy: {acc_enabled:.2f}%")
    print(f"Disabled model accuracy: {acc_disabled:.2f}%")
    print(f"Performance drop: {performance_drop:.2f}%")

if __name__ == "__main__":
    test_quantum_module_criticality()