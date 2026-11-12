"""Predict kinship for two face-cropped photos using a trained VQC checkpoint."""
import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL = os.path.join(
    PROJECT_ROOT, "weights", "deploy", "amplitude_vqc.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Predict kin/non-kin using the joint-register VQC experiment."
    )
    parser.add_argument("--img1", required=True, help="first face-cropped photo")
    parser.add_argument("--img2", required=True, help="second face-cropped photo")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="checkpoint from scripts/training/train_amplitude_vqc.py")
    args = parser.parse_args()

    for path in (args.img1, args.img2):
        if not os.path.isfile(path):
            parser.error(f"image not found: {path}")
    if not os.path.isfile(args.model):
        parser.error(
            f"VQC checkpoint not found: {args.model}\n"
            "Train it first with: "
            "python scripts/training/train_amplitude_vqc.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if config.get("readout") != "both":
        parser.error("checkpoint is not a dual-readout VQC model trained by this workflow")
    if config.get("backbone") != "facenet":
        parser.error(
            f"checkpoint uses {config.get('backbone')} embeddings; "
            "this image script currently supports FaceNet checkpoints only"
        )

    from src.models_vqc import AmplitudeVQCClassifier
    from src.user_input import extract_faces

    model = AmplitudeVQCClassifier(
        n_qubits_each=int(config["n_qubits_each"]),
        depth=int(config["depth"]),
        dropout=float(config["dropout"]),
        readout="both",
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    embeddings, failures = extract_faces([args.img1, args.img2], detect=False)
    if failures or len(embeddings) != 2:
        detail = "\n".join(f"  {path}: {reason}" for path, reason in failures)
        parser.error(f"Could not embed both photos.\n{detail}")

    emb1 = torch.as_tensor(np.asarray(embeddings[0]), dtype=torch.float32,
                           device=device).unsqueeze(0)
    emb2 = torch.as_tensor(np.asarray(embeddings[1]), dtype=torch.float32,
                           device=device).unsqueeze(0)
    relations = torch.zeros((1, 4), dtype=torch.float32, device=device)
    with torch.no_grad():
        probability = float(model.predict(emb1, emb2, relations).item())

    threshold = float(checkpoint["threshold"])
    metrics = checkpoint.get("metrics", {})
    print(f"Kin probability : {probability:.4f}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Prediction      : {'KIN' if probability >= threshold else 'NON-KIN'}")
    if metrics:
        print("Validation      : %.2f%% accuracy, ROC-AUC %.4f"
              % (metrics["validation_accuracy"],
                 metrics["validation_roc_auc"]))
    if abs(probability - threshold) < 0.08:
        print("Caution         : borderline score; treat as uncertain.")
    print("Research prototype: this score is not proof of biological relatedness.")


if __name__ == "__main__":
    main()
