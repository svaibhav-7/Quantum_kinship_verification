"""Train a deployable, dual-readout amplitude-encoded VQC checkpoint.

The split is family-disjoint: validation families are excluded from training.
The checkpoint threshold is calibrated on that validation split.
"""
import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import prepare_pair_tensors
from src.kfold import grouped_kfold
from src.models_vqc import AmplitudeVQCClassifier
from scripts.evaluation.run_amplitude_vqc import CACHES, add_neg, keyfn_for, load_positives


def _usable_pairs(pairs, cache):
    return [
        pair for pair in pairs
        if os.path.normcase(os.path.abspath(pair[0])) in cache
        and os.path.normcase(os.path.abspath(pair[1])) in cache
    ]


def _scores(model, tensors, batch_size):
    model.eval()
    e1, e2, _labels, relations = tensors
    out = []
    with torch.no_grad():
        for start in range(0, len(e1), batch_size):
            stop = start + batch_size
            out.append(model.predict(e1[start:stop], e2[start:stop],
                                     relations[start:stop]).view(-1).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Train the diagram-style VQC for two-image kinship prediction."
    )
    parser.add_argument("--dataset", choices=("KinFaceW-I", "KinFaceW-II", "FIW",
                                               "TSKinFace"),
                        default="KinFaceW-I")
    parser.add_argument("--backbone", choices=sorted(CACHES), default="facenet")
    parser.add_argument("--qubits", type=int, default=6,
                        help="qubits per person; the circuit uses twice this number")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--cap-per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "weights", "deploy", "amplitude_vqc.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = CACHES[args.backbone]
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"embedding cache not found: {cache_path}. Build the project caches first."
        )
    with open(cache_path, "rb") as f:
        cache = {
            os.path.normcase(os.path.abspath(path)): embedding
            for path, embedding in pickle.load(f).items()
        }

    datasets = load_positives(args.seed, args.cap_per_family)
    positives = _usable_pairs(datasets.get(args.dataset, []), cache)
    if not positives:
        raise RuntimeError(
            f"No cached positive pairs found for {args.dataset}. "
            "Check that the dataset and matching embedding cache are available."
        )

    keyfn = keyfn_for(args.dataset)
    outer = list(grouped_kfold(positives, keyfn, n_splits=5, seed=args.seed))
    if not outer:
        raise RuntimeError("Not enough family groups to create a validation split.")
    train_positive, _heldout_positive = outer[0]
    inner = list(grouped_kfold(train_positive, keyfn, n_splits=5,
                               seed=args.seed + 7))
    if not inner:
        raise RuntimeError("Not enough training families to create train/validation sets.")
    fit_positive, validation_positive = inner[0]

    fit_pairs = _usable_pairs(
        add_neg(fit_positive, keyfn, args.seed + 1), cache)
    validation_pairs = _usable_pairs(
        add_neg(validation_positive, keyfn, args.seed + 2), cache)
    if len(fit_pairs) < 200 or len(validation_pairs) < 60:
        raise RuntimeError(
            f"Insufficient usable pairs after splitting: "
            f"{len(fit_pairs)} train, {len(validation_pairs)} validation."
        )

    train = tuple(t.to(device) for t in prepare_pair_tensors(fit_pairs, cache))
    validation = tuple(
        t.to(device) for t in prepare_pair_tensors(validation_pairs, cache))
    validation_labels = validation[2].view(-1).cpu().numpy()
    if len(np.unique(validation_labels)) != 2:
        raise RuntimeError("Validation split must contain kin and non-kin pairs.")

    model = AmplitudeVQCClassifier(
        n_qubits_each=args.qubits,
        depth=args.depth,
        dropout=args.dropout,
        readout="both",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_auc, best_state, epochs_without_improvement = -1.0, None, 0

    print(f"Training dual-readout VQC on {args.dataset} ({device})")
    print(f"Pairs: {len(train[2])} train, {len(validation[2])} validation")
    print(f"Circuit: {args.qubits * 2} qubits, depth {args.depth}")
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(train[2]), device=device)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            optimizer.zero_grad()
            logits = model.forward_logits(
                train[0][indices], train[1][indices], train[3][indices])
            loss = loss_fn(logits, train[2][indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation_scores = _scores(model, validation, args.batch_size)
        auc = float(roc_auc_score(validation_labels, validation_scores))
        print(f"  epoch {epoch + 1:02d}: validation ROC-AUC {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a usable model checkpoint.")
    model.load_state_dict(best_state)
    validation_scores = _scores(model, validation, args.batch_size)
    threshold = calibrate_threshold(
        validation_labels, validation_scores, objective="accuracy", max_fpr=0.25,
        smooth=True,
    )
    metrics = operating_point(validation_labels, validation_scores, threshold)
    checkpoint = {
        "state_dict": best_state,
        "threshold": threshold,
        "metrics": {
            "validation_accuracy": metrics["accuracy"],
            "validation_roc_auc": best_auc,
            "validation_fpr": metrics["fpr"],
        },
        "config": {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "n_qubits_each": args.qubits,
            "depth": args.depth,
            "dropout": args.dropout,
            "readout": "both",
            "preprocessing": "uncropped",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(checkpoint, args.out)
    print(f"Saved checkpoint: {args.out}")
    print(f"Validation accuracy: {metrics['accuracy']:.2f}%  "
          f"ROC-AUC: {best_auc:.4f}  threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
