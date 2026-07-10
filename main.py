"""
Quantum SWAP Test Kinship Verification -- Interactive Demo (v3)

Demonstrates product-state and shared-circuit encoding modes
using the trained HybridKinshipClassifier.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path so 'src' can be resolved by IDEs and when running from subdirectories
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import FaceFeatureExtractor, HybridKinshipClassifier
from src.quantum_core import simulate_swap_test, simulate_entangled_swap_test
from src.data_loaders import get_relation_category


def load_model(
    weights_path, encoding_mode="entangled", projection_type="cross_attention"
):
    """Load the trained model, auto-detecting config from checkpoint."""
    n_qubits = 8

    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

            # Detect n_qubits from projection output layer
            for key in state_dict:
                if key.endswith(".bias") and "projection" in key:
                    candidate = state_dict[key].shape[0]
                    if candidate <= 16:  # Reasonable qubit count
                        n_qubits = candidate

            # Detect encoding mode from presence of legacy or new entanglement params
            if "ent_params" in state_dict:
                encoding_mode = "entangled"
                n_qubits = state_dict["ent_params"].shape[0]
            elif "ent_params1" in state_dict and "ent_params2" in state_dict:
                encoding_mode = "entangled"
                n_qubits = state_dict["ent_params1"].shape[0]

            # Detect projection type from key names
            if any("cross_attn" in k for k in state_dict):
                projection_type = "cross_attention"
            elif any("projection.projection" in k for k in state_dict):
                projection_type = "simple"

            print(
                f"  [INFO] Detected: n_qubits={n_qubits}, "
                f"encoding={encoding_mode}, projection={projection_type}"
            )

            model = HybridKinshipClassifier(
                n_qubits=n_qubits,
                encoding_mode=encoding_mode,
                projection_type=projection_type,
            )
            model.load_state_dict(state_dict)
            print(f"  [OK] Loaded weights from: {weights_path}")
            return model

        except Exception as e:
            print(f"  [WARN] Failed to load weights ({e}). Using random init.")
    else:
        print("  [INFO] No trained weights found. Using random initialization.")

    return HybridKinshipClassifier(
        n_qubits=n_qubits,
        encoding_mode=encoding_mode,
        projection_type=projection_type,
    )


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(project_root, "weights", "hybrid_kinship_entangled.pt")
    KFW1 = os.path.join(project_root, "KinFaceW-I", "KinFaceW-I")

    print("=" * 64)
    print("    QUANTUM SWAP TEST KINSHIP VERIFICATION DEMO (v3)")
    print("=" * 64)

    # 1. Initialize models
    print("\n[1/3] Initializing models...")
    extractor = FaceFeatureExtractor()
    model = load_model(weights_path)
    model.eval()

    encoding_mode = model.encoding_mode
    print(f"  Encoding mode: {encoding_mode}")
    print("-" * 64)

    # 2. Load a sample pair from KinFaceW-I
    print("\n[2/3] Loading sample pair from KinFaceW-I...")

    sample_dir = os.path.join(KFW1, "images", "father-dau")
    if not os.path.exists(sample_dir):
        print(f"  [ERROR] Dataset directory not found: {sample_dir}")
        print("  Please download datasets and place them in the project root.")
        return

    images = sorted([img for img in os.listdir(sample_dir) if img.endswith(".jpg")])
    if len(images) < 2:
        print("  [ERROR] Not enough images in father-dau directory.")
        return

    p1 = os.path.join(sample_dir, images[0])
    p2 = os.path.join(sample_dir, images[1])

    print(f"  Face 1: {os.path.basename(p1)}")
    print(f"  Face 2: {os.path.basename(p2)}")
    print("-" * 64)

    # 3. Perform Kinship Verification
    print("\n[3/3] Running kinship predictions...")
    try:
        # Build relation tensor
        rel_type = "father-dau"
        cat = get_relation_category(rel_type)
        one_hot = [0.0] * 4
        one_hot[cat] = 1.0
        rel_tensor = torch.tensor([one_hot], dtype=torch.float32)

        # Extract embeddings
        emb1 = extractor.extract(p1)
        emb2 = extractor.extract(p2)
        emb1_t = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
        emb2_t = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)

        # Model forward pass (entangled or product, depending on model config)
        with torch.no_grad():
            prob_model = model(emb1_t, emb2_t, rel_tensor).item()

        # Qiskit circuit verification
        with torch.no_grad():
            z1, z2 = model.projection_net(emb1_t, emb2_t, rel_tensor)
            z1_np = z1.squeeze().numpy()
            z2_np = z2.squeeze().numpy()

        if encoding_mode == "entangled":
            ent1_np = model.ent_params1.detach().cpu().numpy()
            ent2_np = model.ent_params2.detach().cpu().numpy()
            prob_qiskit = simulate_entangled_swap_test(
                z1_np,
                z2_np,
                ent_params1=ent1_np,
                ent_params2=ent2_np,
                shots=1024,
            )
        else:
            prob_qiskit = simulate_swap_test(z1_np, z2_np, shots=1024)

        decision_model = (
            "RELATED (KIN)" if prob_model >= 0.5 else "NOT RELATED (NON-KIN)"
        )
        decision_qiskit = (
            "RELATED (KIN)" if prob_qiskit >= 0.5 else "NOT RELATED (NON-KIN)"
        )

        print(f"\n  Prediction Results ({encoding_mode} encoding):")
        print(f"  * Model Fidelity:       {prob_model * 100:.2f}%  ({decision_model})")
        print(
            f"  * Qiskit SWAP Test:     {prob_qiskit * 100:.2f}%  ({decision_qiskit})"
        )

        if encoding_mode == "entangled":
            # The current shared CNOT/Rz circuit preserves fidelity, so this
            # should match the model value up to floating-point / shot noise.
            cos_diff = np.cos((z1_np - z2_np) / 2.0)
            product_fidelity = np.prod(cos_diff**2)
            print(
                f"  * Product-state cos²:   {product_fidelity * 100:.2f}%  "
                f"(shared-unitary equivalent)"
            )
            delta = abs(prob_model - product_fidelity) * 100
            print(f"  * Circuit vs Product delta: {delta:.2f} percentage points")

    except Exception as e:
        print(f"  [ERROR] Inference failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
