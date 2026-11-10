#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kinship verification for your own photos.

Run with no arguments for an interactive prompt:

    python verify_kinship.py

Or pass photos directly. Each person may be given as one photo, several
photos, a folder, or a URL:

    python verify_kinship.py --person-a mum.jpg --person-b kid.jpg
    python verify_kinship.py --person-a ./mum_photos/ --person-b kid1.jpg kid2.jpg
    python verify_kinship.py --person-a https://site/a.jpg --person-b b.jpg

Triadic mode, when both parents are available (more accurate):

    python verify_kinship.py --father f.jpg --mother m.jpg --child c.jpg

Show random, held-out dataset examples under protocol v2:

    python verify_kinship.py --dataset-samples --dataset FIW --samples 12
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.user_input import (InputError, confidence_label, extract_faces,
                            resolve_inputs)



def _deployed(name):
    """Prefer the protocol-v2 model; fall back to the earlier one if absent."""
    current = os.path.join(PROJECT_ROOT, "weights", "deploy", "v2", name)
    return current if os.path.exists(current) else os.path.join(
        PROJECT_ROOT, "weights", "deploy", name)


SET_MODEL = _deployed("set_model.pkl")
TRIAD_MODEL = _deployed("triad_model.pkl")
CACHE_PATH = os.path.join(PROJECT_ROOT, "weights", "caches", "all_datasets_cache.pkl")
DATASET_NAMES = ("FIW", "KinFaceW-I", "KinFaceW-II", "TSKinFace")

BAR = "=" * 62


def _sample_balanced_indices(labels, count, seed):
    """Choose a reproducible, near-balanced sample from labels 0 and 1."""
    labels = np.asarray(labels)
    if count < 1:
        raise ValueError("sample count must be positive")
    if count > len(labels):
        raise ValueError(f"requested {count} samples but only {len(labels)} are available")
    by_label = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    per_class = count // 2
    available = min(len(by_label[0]), len(by_label[1]))
    if per_class > available:
        raise ValueError(
            f"requested {count} samples but only {available} examples per class "
            "are available in this held-out fold")

    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        selected.extend(rng.choice(by_label[label], per_class, replace=False).tolist())
    if count % 2:
        used = set(selected)
        remaining = np.asarray([i for i in range(len(labels)) if i not in used])
        selected.append(int(rng.choice(remaining)))
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _load_dataset_positives(dataset, cache, seed):
    """Load positive pairs only; protocol negatives are generated after splitting."""
    from src.data_loaders import load_fiw_pairs, load_kinfacew_pairs
    from src.fair_protocol import drop_duplicate_groups
    from src.kfold import balance_group_sizes
    from src.multi_dataset import _kfw_family
    from src.splits import family_of
    from src.ts_pairs import build_tskinface_pairs, family_of_ts

    if dataset == "FIW":
        rows = [row for row in load_fiw_pairs(os.path.join(PROJECT_ROOT, "public"))
                if int(row[2]) == 1]
        rows = balance_group_sizes(rows, family_of, 1500, seed)
        keyfn = family_of
    elif dataset in ("KinFaceW-I", "KinFaceW-II"):
        root = os.path.join(PROJECT_ROOT, dataset)
        rows = [row[:4] for row in load_kinfacew_pairs(root) if int(row[2]) == 1]
        keyfn = _kfw_family
    elif dataset == "TSKinFace":
        root = os.path.join(PROJECT_ROOT, "TSKinFace_Data", "TSKinFace_Data",
                            "TSKinFace_cropped")
        rows = [row for row in build_tskinface_pairs(root, 0.5, seed)
                if int(row[2]) == 1]
        keyfn = family_of_ts
    else:
        raise ValueError(f"unsupported dataset {dataset!r}")

    rows = [tuple(row[:4]) for row in rows
            if os.path.normcase(os.path.abspath(row[0])) in cache
            and os.path.normcase(os.path.abspath(row[1])) in cache]
    rows, dropped = drop_duplicate_groups(
        rows, keyfn, lambda path: cache[os.path.normcase(os.path.abspath(path))])
    if dropped:
        print(f"Removed duplicate family groups: {', '.join(map(str, dropped))}")
    if not rows:
        raise ValueError(
            f"No cached positive pairs are available for {dataset}; build or update "
            f"{os.path.relpath(CACHE_PATH, PROJECT_ROOT)} first")
    return rows, keyfn


def _cached_pair_tensors(pairs, cache, device):
    import torch

    normalize = lambda path: os.path.normcase(os.path.abspath(path))
    first = torch.as_tensor(np.stack([cache[normalize(pair[0])] for pair in pairs]),
                            dtype=torch.float32, device=device)
    second = torch.as_tensor(np.stack([cache[normalize(pair[1])] for pair in pairs]),
                             dtype=torch.float32, device=device)
    labels = torch.as_tensor([int(pair[2]) for pair in pairs],
                             dtype=torch.float32, device=device)
    return first, second, labels


def _train_demo_model(fit, validation, seed, fold, device):
    """Train the protocol-v2 fair reference model; choose seed by validation AUC."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    from src.models_fair import FairKinshipModel

    best_auc, best_state = -1.0, None
    for model_seed in range(3):
        torch.manual_seed(seed + 1000 * fold + model_seed)
        model = FairKinshipModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        run_best, best_epoch_state, patience = -1.0, None, 0

        for _epoch in range(40):
            model.train()
            order = torch.randperm(len(fit[2]), device=device)
            for start in range(0, len(order), 512):
                indices = order[start:start + 512]
                if len(indices) < 2:
                    continue
                optimizer.zero_grad()
                logits = model.forward_logits(fit[0][indices], fit[1][indices]).view(-1)
                loss = loss_fn(logits, fit[2][indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_scores = model(validation[0], validation[1]).view(-1).cpu().numpy()
            val_auc = float(roc_auc_score(validation[2].cpu().numpy(), val_scores))
            if val_auc > run_best:
                run_best = val_auc
                patience = 0
                best_epoch_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                patience += 1
                if patience >= 8:
                    break

        if run_best > best_auc and best_epoch_state is not None:
            best_auc, best_state = run_best, best_epoch_state

    if best_state is None:
        raise RuntimeError("No validation-selected model state was produced")
    selected = FairKinshipModel().to(device)
    selected.load_state_dict(best_state)
    selected.eval()
    return selected, best_auc


def _plot_dataset_samples(dataset, pairs, indices, scores, threshold, fold,
                          accuracy, output_path, seed):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    columns = 3
    rows = math.ceil(len(indices) / columns)
    figure = plt.figure(figsize=(15, rows * 3.2 + 1.7), constrained_layout=True)
    grid = figure.add_gridspec(rows, columns)
    for sample_number, pair_index in enumerate(indices):
        row, column = divmod(sample_number, columns)
        cell = grid[row, column].subgridspec(1, 2, wspace=0.03)
        pair = pairs[int(pair_index)]
        probability = float(scores[int(pair_index)])
        truth = int(pair[2])
        predicted = int(probability >= threshold)
        correct = truth == predicted
        edge = "#22863a" if correct else "#cb2431"
        relation = str(pair[3]).upper()
        title = (
            f"S{sample_number + 1} | {relation}\n"
            f"True: {'KIN' if truth else 'NON-KIN'}  |  "
            f"Pred: {'KIN' if predicted else 'NON-KIN'}  |  "
            f"{'CORRECT' if correct else 'WRONG'}\n"
            f"Score: {probability:.1%}  (threshold {threshold:.1%})"
        )
        for endpoint, image_path in enumerate(pair[:2]):
            axis = figure.add_subplot(cell[0, endpoint])
            with Image.open(image_path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel(os.path.basename(image_path), fontsize=7)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(edge)
                spine.set_linewidth(2.5)
            if endpoint == 0:
                axis.set_title(title, fontsize=9, color=edge, fontweight="bold")

    figure.suptitle(
        f"PROTOCOL V2 — {dataset} held-out fold {fold + 1} (seed {seed}) | "
        f"Random examples: {accuracy}/{len(indices)} correct\n"
        "Family-disjoint test pairs; relation-matched negatives; "
        "corpus-specific model and threshold selected on validation only",
        fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_dataset_samples(dataset, sample_count, seed, fold, output_path):
    """Train on protocol-v2 fit/validation families and visualize random test pairs."""
    import torch
    from sklearn.metrics import roc_auc_score

    from src.calibration import calibrate_threshold
    from src.fair_protocol import build_side, split_fit_val
    from src.kfold import grouped_kfold

    if not os.path.isfile(CACHE_PATH):
        raise FileNotFoundError(
            f"embedding cache not found: {CACHE_PATH}. Build it before running this demo.")
    if fold < 1 or fold > 5:
        raise ValueError("fold must be between 1 and 5")

    with open(CACHE_PATH, "rb") as stream:
        raw_cache = pickle.load(stream)
    cache = {os.path.normcase(os.path.abspath(path)): np.asarray(vector, dtype=np.float32)
             for path, vector in raw_cache.items()}
    positives, keyfn = _load_dataset_positives(dataset, cache, seed)
    folds = list(grouped_kfold(positives, keyfn, n_splits=5, seed=42))
    if len(folds) != 5:
        raise ValueError(
            f"{dataset} produced {len(folds)} usable family folds; five are required")
    train_positive, test_positive = folds[fold - 1]
    fit_positive, validation_positive = split_fit_val(
        train_positive, keyfn, seed=42 + fold)
    fit_pairs = build_side(fit_positive, keyfn, seed + fold)
    validation_pairs = build_side(validation_positive, keyfn, seed + 100 + fold)
    test_pairs = build_side(test_positive, keyfn, seed + 500 + fold)
    for name, pairs in (("fit", fit_pairs), ("validation", validation_pairs),
                        ("test", test_pairs)):
        if not pairs or {int(pair[2]) for pair in pairs} != {0, 1}:
            raise ValueError(
                f"{dataset} fold {fold}: {name} split lacks one class after "
                "relation-matched negatives. Try another --fold or --seed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = _cached_pair_tensors(fit_pairs, cache, device)
    validation = _cached_pair_tensors(validation_pairs, cache, device)
    test = _cached_pair_tensors(test_pairs, cache, device)
    model, validation_auc = _train_demo_model(fit, validation, seed, fold - 1, device)
    with torch.no_grad():
        validation_scores = model(validation[0], validation[1]).view(-1).cpu().numpy()
        test_scores = model(test[0], test[1]).view(-1).cpu().numpy()
    threshold = calibrate_threshold(
        validation[2].cpu().numpy(), validation_scores, "accuracy",
        max_fpr=0.25, smooth=True)
    labels = test[2].cpu().numpy().astype(np.int8)
    sample_indices = _sample_balanced_indices(labels, sample_count, seed)
    predictions = test_scores[sample_indices] >= threshold
    correct_count = int(np.count_nonzero(predictions == labels[sample_indices]))
    sample_accuracy = correct_count / len(sample_indices) * 100.0
    if output_path is None:
        output_path = os.path.join(
            PROJECT_ROOT, "results", "demo",
            f"kinship_samples_{dataset.lower().replace('-', '_')}_fold{fold}_seed{seed}.png")
    _plot_dataset_samples(dataset, test_pairs, sample_indices, test_scores, threshold,
                          fold - 1, sample_accuracy, output_path, seed)
    print(f"Dataset: {dataset} | held-out family fold: {fold}/5")
    print(f"Fit pairs: {len(fit_pairs)} | validation pairs: {len(validation_pairs)} "
          f"| test pairs: {len(test_pairs)}")
    print(f"Validation AUC (selected model): {validation_auc:.3f}")
    print(f"Validation-calibrated threshold: {threshold:.4f}")
    print(f"Random sample: {len(sample_indices)} pairs, "
          f"{int(predictions.sum())} predicted kin")
    print(f"Sample correct: {correct_count}/"
          f"{len(sample_indices)} ({sample_accuracy:.1f}%)")
    print("This sample is illustrative; its accuracy is not a benchmark estimate.")
    print(f"Saved visualization: {output_path}")


def detect_default_for(predictor):
    """Face detection must match how the model was fitted.

    The embeddings a model was fitted on and the embeddings it is served must
    be produced the same way. Measured on 387 person-pairs, serving
    MTCNN-cropped embeddings to a model fitted on uncropped ones flipped
    30.2% of verdicts (mean score shift 0.20) even though aggregate ROC-AUC
    looked unchanged -- which is why aggregate AUC is the wrong check.
    """
    return getattr(predictor, "preprocessing", "uncropped") == "cropped"


def _rule(title=""):
    print(BAR if not title else f"\n{title}\n" + "-" * 62)


def _gather(label, raw, detect):
    """Resolve inputs, embed, and report anything unusable."""
    paths = resolve_inputs(raw)
    embs, failed = extract_faces(paths, detect=detect)

    print(f"  {label}: {len(embs)} usable photo(s) of {len(paths)} supplied")
    for p, why in failed:
        print(f"    skipped {os.path.basename(p)} -- {why}")

    if not embs:
        raise InputError(
            f"no usable face found for {label}. "
            "Try a clearer, front-facing photo, or pass --no-detect if the "
            "images are already tight face crops.")
    return np.stack(embs)


def _verdict(prob, threshold, metrics):
    conf = confidence_label(prob, threshold)
    kin = prob >= threshold

    _rule("RESULT")
    print(f"  score      : {prob:.4f}   (decides KIN at >= {threshold:.4f})")
    print(f"  verdict    : {'RELATED (KIN)' if kin else 'NOT RELATED'}")
    print(f"  confidence : {conf}")

    if conf == "borderline":
        print("\n  This score sits close to the decision boundary. Treat it as")
        print("  inconclusive rather than as an answer.")

    acc = metrics.get("accuracy")
    if acc:
        print(f"\n  For context: this model is right about {acc:.0f}% of the time")
        print("  on held-out families it never saw in training. It is a research")
        print("  prototype -- do not use it for any decision that affects people.")


def run_pair(a_raw, b_raw, detect):
    from src.set_predictor import SetKinshipPredictor

    if not os.path.exists(SET_MODEL):
        sys.exit("model missing. Run: python scripts/deploy/package_setlevel.py")

    pred = SetKinshipPredictor(SET_MODEL)
    if detect is None:
        detect = detect_default_for(pred)
    _rule("PHOTOS")
    print(f"  model was fitted on {pred.preprocessing} faces; "
          f"detection {'ON' if detect else 'OFF'}")
    A = _gather("person A", a_raw, detect)
    B = _gather("person B", b_raw, detect)

    r = pred.predict_sets(A, B)
    _verdict(r["probability"], pred.threshold, pred.metrics)

    if r["n_a"] == 1 and r["n_b"] == 1:
        print("\n  Tip: one photo each was used. Several photos per person is")
        print("  measurably more reliable -- pass a folder or multiple files.")


def run_triad(f_raw, m_raw, c_raw, detect):
    from src.set_predictor import TriadPredictor

    if not os.path.exists(TRIAD_MODEL):
        sys.exit("model missing. Run: python scripts/deploy/package_setlevel.py")

    pred = TriadPredictor(TRIAD_MODEL)
    if detect is None:
        detect = detect_default_for(pred)
    _rule("PHOTOS")
    print(f"  model was fitted on {pred.preprocessing} faces; "
          f"detection {'ON' if detect else 'OFF'}")
    F = _gather("father", f_raw, detect).mean(axis=0)
    M = _gather("mother", m_raw, detect).mean(axis=0)
    C = _gather("child", c_raw, detect).mean(axis=0)

    r = pred.predict_triad(F, M, C)
    _verdict(r["probability"], pred.threshold, pred.metrics)
    print(f"\n  resembles  : {r['resembles']} more (alpha = {r['alpha']:.2f})")
    print(f"  similarity : father {r['cos_fc']:.3f}   mother {r['cos_mc']:.3f}")


def _ask(prompt):
    """Read a whitespace-separated list of photos from the user."""
    raw = input(prompt).strip()
    if not raw:
        return []
    # Keep quoted paths with spaces intact.
    import shlex

    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def interactive(detect):
    print(BAR)
    print("  KINSHIP VERIFICATION")
    print(BAR)
    print("\n  Give photos of two people and I will estimate whether they are")
    print("  biologically related. For each person you may enter:")
    print("    - one photo        mum.jpg")
    print("    - several photos   a.jpg b.jpg c.jpg   (more is better)")
    print("    - a folder         ./mum_photos/")
    print("    - a URL            https://site/photo.jpg")
    print("\n  Enter 't' for triadic mode (father + mother + child).")

    first = _ask("\n  Person A photos (or 't'): ")
    if len(first) == 1 and first[0].lower() == "t":
        f = _ask("  Father photo(s): ")
        m = _ask("  Mother photo(s): ")
        c = _ask("  Child photo(s) : ")
        if not (f and m and c):
            sys.exit("  all three are required for triadic mode")
        run_triad(f, m, c, detect)
        return

    b = _ask("  Person B photos: ")
    if not (first and b):
        sys.exit("  photos are required for both people")
    run_pair(first, b, detect)


def main():
    ap = argparse.ArgumentParser(
        description="Estimate whether two people are biologically related.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--person-a", nargs="+", metavar="PHOTO",
                    help="photo(s), folder or URL for the first person")
    ap.add_argument("--person-b", nargs="+", metavar="PHOTO",
                    help="photo(s), folder or URL for the second person")
    ap.add_argument("--father", nargs="+", metavar="PHOTO")
    ap.add_argument("--mother", nargs="+", metavar="PHOTO")
    ap.add_argument("--child", nargs="+", metavar="PHOTO")
    ap.add_argument("--dataset-samples", action="store_true",
                    help="train a protocol-v2-style reference on one dataset fold and "
                         "save a visualization of random held-out pairs")
    ap.add_argument("--dataset", choices=DATASET_NAMES, default="FIW",
                    help="dataset for --dataset-samples (default: FIW)")
    ap.add_argument("--samples", type=int, default=12,
                    help="number of balanced random test examples to visualize")
    ap.add_argument("--fold", type=int, choices=range(1, 6), default=1,
                    help="family-disjoint held-out fold, 1-5 (default: 1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for pair selection, negative sampling and training")
    ap.add_argument("--output-image",
                    help="output PNG path (default: results/demo/kinship_samples_*.png)")
    ap.add_argument("--detect", dest="detect", action="store_true",
                    default=None,
                    help="force face detection on")
    ap.add_argument("--no-detect", dest="detect", action="store_false",
                    help="force face detection off")
    args = ap.parse_args()
    # Left unset, the tool follows whatever the model was fitted on.
    detect = args.detect

    try:
        if args.dataset_samples:
            if (args.person_a or args.person_b or args.father or args.mother or args.child):
                ap.error("--dataset-samples cannot be combined with photo or triad inputs")
            if args.samples < 2:
                ap.error("--samples must be at least 2")
            run_dataset_samples(args.dataset, args.samples, args.seed, args.fold,
                                args.output_image)
        elif args.father or args.mother or args.child:
            if not (args.father and args.mother and args.child):
                sys.exit("triadic mode needs --father, --mother and --child")
            run_triad(args.father, args.mother, args.child, detect)
        elif args.person_a or args.person_b:
            if not (args.person_a and args.person_b):
                sys.exit("both --person-a and --person-b are required")
            run_pair(args.person_a, args.person_b, detect)
        else:
            interactive(detect)
    except InputError as e:
        sys.exit(f"\n  error: {e}")
    except KeyboardInterrupt:
        sys.exit("\n  cancelled")


if __name__ == "__main__":
    main()
