"""Reconstructed formulation reruns on corrected, family-disjoint TSKinFace.

The original implementations for formulations 1-7 are not present in this
checkout, and the manuscript leaves several operational details unspecified.
This runner records those choices in its output and does not present these
runs as exact replications of the historical experiments.
"""
import argparse
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.calibration import calibrate_threshold
from src.data_loaders import get_relation_category, prepare_pair_tensors
from src.identity_sets import features_to_vector, pair_set_features
from src.kfold import grouped_kfold
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.quantum_fast import _cnot_permutation, _rz_phases, fast_entangled_fidelity
from src.ts_pairs import (
    add_tskinface_negatives,
    build_tskinface_pairs,
    family_of_ts,
)
from src.triads import TRIAD_FEATURE_ORDER, find_triads, triad_features

TS_ROOT = os.path.join(
    PROJECT_ROOT, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped"
)
CACHE_PATH = os.path.join(
    PROJECT_ROOT, "weights", "caches", "all_datasets_cache.pkl"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "honest")
RELATIONS = ("fd", "fs", "md", "ms")


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def load_cache():
    with open(CACHE_PATH, "rb") as stream:
        raw = pickle.load(stream)
    return {_norm(path): np.asarray(value, dtype=np.float32)
            for path, value in raw.items()}


def load_positive_pairs(cache, seed):
    pairs = [
        pair for pair in build_tskinface_pairs(TS_ROOT, 0.5, seed)
        if int(pair[2]) == 1
    ]
    if not pairs:
        raise RuntimeError("No cached positive TSKinFace pairs were found.")
    missing = {
        _norm(path)
        for pair in pairs
        for path in pair[:2]
        if _norm(path) not in cache
    }
    if missing:
        raise RuntimeError(
            "%d positive-pair image paths are missing cached embeddings."
            % len(missing)
        )
    return pairs


def corrected_side(positive_pairs, seed):
    pairs = add_tskinface_negatives(
        positive_pairs,
        keyfn=family_of_ts,
        seed=seed,
        same_family_negative_ratio=0.5,
    )
    labels = [int(pair[2]) for pair in pairs]
    if labels.count(1) != labels.count(0):
        raise RuntimeError("Corrected TSKinFace side is not class balanced.")
    return pairs


def split_families(pairs):
    return {family_of_ts(pair[0]) for pair in pairs}


def assert_side_is_closed(pairs, allowed_families):
    for pair in pairs:
        if family_of_ts(pair[0]) not in allowed_families:
            raise RuntimeError("Pair endpoint escaped its assigned split.")
        if family_of_ts(pair[1]) not in allowed_families:
            raise RuntimeError("Pair endpoint escaped its assigned split.")


def make_outer_folds(positive_pairs, seed, folds=5):
    result = []
    for index, (train, test) in enumerate(
        grouped_kfold(positive_pairs, family_of_ts, folds, seed)
    ):
        train_families = split_families(train)
        test_families = split_families(test)
        if train_families & test_families:
            raise RuntimeError("Outer TSKinFace folds share a family.")
        train_side = corrected_side(train, seed + 100 + index)
        test_side = corrected_side(test, seed + 500 + index)
        assert_side_is_closed(train_side, train_families)
        assert_side_is_closed(test_side, test_families)
        result.append({
            "index": index,
            "train_positive": train,
            "test_positive": test,
            "train": train_side,
            "test": test_side,
            "train_families": train_families,
            "test_families": test_families,
        })
    if len(result) != folds:
        raise RuntimeError("Expected %d usable family folds; got %d."
                           % (folds, len(result)))
    return result


def make_inner_sides(train_positive, seed):
    folds = list(grouped_kfold(train_positive, family_of_ts, 5, seed))
    if not folds:
        raise RuntimeError("Could not make a family-disjoint validation split.")
    fit_positive, validation_positive = folds[0]
    fit_families = split_families(fit_positive)
    validation_families = split_families(validation_positive)
    if fit_families & validation_families:
        raise RuntimeError("Inner fit and validation families overlap.")
    fit = corrected_side(fit_positive, seed + 1000)
    validation = corrected_side(validation_positive, seed + 2000)
    assert_side_is_closed(fit, fit_families)
    assert_side_is_closed(validation, validation_families)
    return fit, validation


def metric_row(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "accuracy": float(accuracy_score(labels, scores >= threshold)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "test_labels": labels.tolist(),
        "test_scores": scores.tolist(),
    }


def add_pair_families(row, pairs):
    family_a = [family_of_ts(pair[0]) for pair in pairs]
    family_b = [family_of_ts(pair[1]) for pair in pairs]
    if len(family_a) != len(row["test_labels"]):
        raise ValueError("Prediction rows do not match test pair families.")
    row["test_family_a"] = family_a
    row["test_family_b"] = family_b


def _weighted_auc_plan(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    return labels[order], order, starts


def _weighted_auc(plan, weights):
    sorted_labels, order, starts = plan
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    positive = np.add.reduceat(
        sorted_weights * (sorted_labels == 1), starts
    )
    negative = np.add.reduceat(
        sorted_weights * (sorted_labels == 0), starts
    )
    total_positive = positive.sum()
    total_negative = negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return None
    negative_before = np.cumsum(negative) - negative
    return float(
        np.sum(positive * (negative_before + 0.5 * negative))
        / (total_positive * total_negative)
    )


def cluster_bootstrap_delta(treatment, control, seed, replicates=2000):
    """Bootstrap paired AUC differences by family clusters within each fold."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    fields = ("test_labels", "test_scores", "test_family_a", "test_family_b")
    for arm_name, rows in (("treatment", treatment), ("control", control)):
        if not rows or any(any(field not in row for field in fields) for row in rows):
            raise ValueError("%s rows are missing saved test predictions." % arm_name)

    labels = np.concatenate([
        np.asarray(row["test_labels"], dtype=np.int64) for row in treatment
    ])
    treatment_scores = np.concatenate([
        np.asarray(row["test_scores"], dtype=np.float64) for row in treatment
    ])
    control_scores = np.concatenate([
        np.asarray(row["test_scores"], dtype=np.float64) for row in control
    ])
    family_a = np.concatenate([
        np.asarray(row["test_family_a"], dtype=object) for row in treatment
    ])
    family_b = np.concatenate([
        np.asarray(row["test_family_b"], dtype=object) for row in treatment
    ])
    control_labels = np.concatenate([
        np.asarray(row["test_labels"], dtype=np.int64) for row in control
    ])
    control_family_a = np.concatenate([
        np.asarray(row["test_family_a"], dtype=object) for row in control
    ])
    control_family_b = np.concatenate([
        np.asarray(row["test_family_b"], dtype=object) for row in control
    ])
    if not (
        np.array_equal(labels, control_labels)
        and np.array_equal(family_a, control_family_a)
        and np.array_equal(family_b, control_family_b)
        and len(labels) == len(treatment_scores) == len(control_scores)
    ):
        raise ValueError("Treatment and control must score identical test pairs.")

    fold_ids = np.concatenate([
        np.full(len(row["test_labels"]), row["fold"], dtype=np.int64)
        for row in treatment
    ])
    fold_groups = {}
    fold_positions = {}
    auc_plans = {}
    for fold in sorted(set(fold_ids.tolist())):
        positions = np.flatnonzero(fold_ids == fold)
        fold_positions[fold] = positions
        fold_groups[fold] = sorted(
            set(family_a[positions]) | set(family_b[positions])
        )
        auc_plans[fold] = (
            _weighted_auc_plan(labels[positions], treatment_scores[positions]),
            _weighted_auc_plan(labels[positions], control_scores[positions]),
        )
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        fold_deltas = []
        for fold, groups in fold_groups.items():
            sampled = rng.choice(groups, size=len(groups), replace=True)
            unique, counts = np.unique(sampled, return_counts=True)
            multiplicities = dict(zip(unique.tolist(), counts.tolist()))
            positions = fold_positions[fold]
            counts_a = np.asarray(
                [multiplicities.get(group, 0) for group in family_a[positions]],
                dtype=np.float64,
            )
            counts_b = np.asarray(
                [multiplicities.get(group, 0) for group in family_b[positions]],
                dtype=np.float64,
            )
            weights = counts_a * counts_b
            same_family = family_a[positions] == family_b[positions]
            weights[same_family] = counts_a[same_family]
            treatment_auc = _weighted_auc(
                auc_plans[fold][0], weights
            )
            control_auc = _weighted_auc(
                auc_plans[fold][1], weights
            )
            if treatment_auc is None or control_auc is None:
                break
            fold_deltas.append(treatment_auc - control_auc)
        if len(fold_deltas) != len(fold_groups):
            continue
        deltas.append(float(np.mean(fold_deltas)))
    if len(deltas) < max(100, replicates // 2):
        raise RuntimeError(
            "Too few valid family-bootstrap samples: %d/%d."
            % (len(deltas), replicates)
        )

    interval = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean_fold_auc_delta": float(np.mean([
            roc_auc_score(
                np.asarray(treatment_row["test_labels"], dtype=np.int64),
                np.asarray(treatment_row["test_scores"], dtype=np.float64),
            )
            - roc_auc_score(
                np.asarray(control_row["test_labels"], dtype=np.int64),
                np.asarray(control_row["test_scores"], dtype=np.float64),
            )
            for treatment_row, control_row in zip(treatment, control)
        ])),
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_sd": float(np.std(deltas, ddof=1)),
        "valid_replicates": len(deltas),
        "requested_replicates": replicates,
        "cluster_unit": "family",
        "pair_weight": (
            "product of endpoint-family multiplicities; within-family pairs "
            "use one family multiplicity"
        ),
        "fold_aggregation": (
            "equal-weight mean of fold-wise paired AUC differences; scores "
            "from different fold models are never pooled"
        ),
        "interpretation": (
            "Paired family-cluster bootstrap on fixed out-of-fold predictions; "
            "conditions on the five fitted models and is exploratory."
        ),
    }


def summarize_rows(rows):
    aucs = np.asarray([row["roc_auc"] for row in rows], dtype=np.float64)
    accuracies = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    return {
        "folds": len(rows),
        "roc_auc_mean": float(aucs.mean()),
        "roc_auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_sd": (
            float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0
        ),
        "per_fold": rows,
    }


def paired_comparison(treatment, control):
    treatment_auc = np.asarray(
        [row["roc_auc"] for row in treatment], dtype=np.float64
    )
    control_auc = np.asarray(
        [row["roc_auc"] for row in control], dtype=np.float64
    )
    if len(treatment_auc) != len(control_auc):
        raise ValueError("Paired comparisons require identical fold counts.")
    delta = treatment_auc - control_auc
    return {
        "delta_mean": float(delta.mean()),
        "delta_sd": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "fold_deltas": delta.tolist(),
        "interpretation": (
            "Descriptive paired-fold differences; folds partition the same "
            "identity pool, so no fold-based p-value is reported."
        ),
    }


def pair_labels(pairs):
    return np.asarray([int(pair[2]) for pair in pairs], dtype=np.int64)


def pair_relation(pair):
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[get_relation_category(pair[3], pair[0])] = 1.0
    return one_hot


def hybrid_logits_and_aux(model, emb1, emb2, rels):
    h1, h2 = model._encode(emb1, emb2, rels)
    z1, z2 = model._angles(h1, h2)
    fidelity = fast_entangled_fidelity(
        z1, z2, model.ent_params1, model.ent_params2, model.n_qubits
    )
    features = torch.cat(
        [
            h1 + h2,
            torch.abs(h1 - h2),
            h1 * h2,
            F.cosine_similarity(h1, h2, dim=1, eps=1e-8).unsqueeze(1),
            rels,
            fidelity,
        ],
        dim=1,
    )
    return model.classifier(features), h1, h2, z1, z2


def state_purity(angles, ent_params, n_qubits=4):
    angles = angles[:, :n_qubits]
    batch = angles.shape[0]
    dim = 2 ** n_qubits
    state = torch.zeros(
        batch, dim, dtype=torch.complex64, device=angles.device
    )
    state[:, 0] = 1.0
    for qubit in range(n_qubits):
        left, right = 2 ** qubit, 2 ** (n_qubits - qubit - 1)
        tensor = state.reshape(batch, left, 2, right)
        c = torch.cos(angles[:, qubit] / 2).to(state.dtype).view(batch, 1, 1)
        s = torch.sin(angles[:, qubit] / 2).to(state.dtype).view(batch, 1, 1)
        zero, one = tensor[:, :, 0, :], tensor[:, :, 1, :]
        state = torch.stack((c * zero - s * one, s * zero + c * one),
                            dim=2).reshape(batch, dim)
    state = state[:, _cnot_permutation(n_qubits, angles.device)]
    state = state * _rz_phases(
        ent_params[:n_qubits], n_qubits, angles.device
    )
    side_dim = 2 ** (n_qubits // 2)
    state = state.reshape(batch, side_dim, -1)
    density = state @ state.conj().transpose(1, 2)
    return density.abs().square().sum(dim=(1, 2)).real


def decorrelation_penalty(h1, h2):
    penalties = []
    for hidden in (h1, h2):
        centered = hidden - hidden.mean(dim=0, keepdim=True)
        normalized = centered / (
            centered.square().mean(dim=0, keepdim=True).sqrt() + 1e-6
        )
        corr = normalized.transpose(0, 1) @ normalized / max(1, len(hidden))
        off_diagonal = corr - torch.diag(torch.diagonal(corr))
        penalties.append(off_diagonal.square().mean())
    return torch.stack(penalties).mean()


def train_hybrid(fit, validation, test, cache, seed, args, device,
                 use_quantum=True, regularizer=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = QuantumAugmentedKinshipClassifier(
        n_qubits=args.qubits,
        dropout=0.5,
        use_quantum=use_quantum,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-2
    )
    loss_fn = nn.BCEWithLogitsLoss()
    fit_tensors = tuple(
        tensor.to(device) for tensor in prepare_pair_tensors(fit, cache)
    )
    val_tensors = tuple(
        tensor.to(device) for tensor in prepare_pair_tensors(validation, cache)
    )
    test_tensors = tuple(
        tensor.to(device) for tensor in prepare_pair_tensors(test, cache)
    )
    val_y = val_tensors[2].view(-1).cpu().numpy().astype(np.int64)
    best_auc, best_state, patience = -1.0, None, 0

    def scores(tensors):
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensors[2]), 1024):
                end = start + 1024
                first, second, _labels, relations = (
                    tensor[start:end] for tensor in tensors
                )
                forward = model(first, second, relations)
                backward = model(second, first, relations)
                chunks.append((0.5 * (forward + backward)).view(-1))
        return torch.cat(chunks).cpu().numpy()

    for _epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(fit_tensors[2]), device=device)
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            optimizer.zero_grad()
            if regularizer is None:
                logits = model.forward_logits(
                    fit_tensors[0][indices],
                    fit_tensors[1][indices],
                    fit_tensors[3][indices],
                )
                loss = loss_fn(logits, fit_tensors[2][indices])
            else:
                logits, h1, h2, z1, z2 = hybrid_logits_and_aux(
                    model,
                    fit_tensors[0][indices],
                    fit_tensors[1][indices],
                    fit_tensors[3][indices],
                )
                loss = loss_fn(logits, fit_tensors[2][indices])
                if regularizer == "purity":
                    purity = 0.5 * (
                        state_purity(z1, model.ent_params1)
                        + state_purity(z2, model.ent_params2)
                    )
                    loss = loss + args.purity_lambda * (1.0 - purity).mean()
                else:
                    loss = loss + args.purity_lambda * decorrelation_penalty(h1, h2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation_scores = scores(val_tensors)
        auc = float(roc_auc_score(val_y, validation_scores))
        if auc > best_auc:
            best_auc = auc
            patience = 0
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Hybrid model did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    validation_scores = scores(val_tensors)
    threshold = calibrate_threshold(
        val_y, validation_scores, "accuracy", max_fpr=0.25
    )
    test_y = test_tensors[2].view(-1).cpu().numpy().astype(np.int64)
    test_scores = scores(test_tensors)
    return metric_row(test_y, test_scores, threshold), {
        "best_validation_auc": best_auc,
        "threshold": float(threshold),
        "epochs_run": _epoch + 1,
    }


def unit(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    return value / (np.linalg.norm(value) + 1e-12)


def state_entropy(angles, n_qubits=4):
    state = np.array([1.0 + 0.0j], dtype=np.complex128)
    for angle in angles[:n_qubits]:
        state = np.kron(
            state,
            np.asarray([math.cos(angle / 2), math.sin(angle / 2)],
                       dtype=np.complex128),
        )
    dim = 2 ** n_qubits
    for control in range(n_qubits - 1):
        target = control + 1
        perm = np.arange(dim)
        for index in range(dim):
            control_bit = (index >> (n_qubits - 1 - control)) & 1
            if control_bit:
                target_bit = (index >> (n_qubits - 1 - target)) & 1
                if target_bit == 0:
                    flipped = index ^ (1 << (n_qubits - 1 - target))
                    perm[index], perm[flipped] = perm[flipped], perm[index]
        state = state[perm]
    matrix = state.reshape(2 ** (n_qubits // 2), -1)
    density = matrix @ matrix.conj().T
    eigenvalues = np.clip(np.linalg.eigvalsh(density).real, 0.0, 1.0)
    positive = eigenvalues[eigenvalues > 1e-12]
    return float(-np.sum(positive * np.log2(positive)))


def fit_angle_projection(pairs, cache, n_components):
    paths = sorted({_norm(path) for pair in pairs for path in pair[:2]})
    values = np.stack([cache[path] for path in paths])
    scaler = StandardScaler().fit(values)
    pca = PCA(n_components=n_components, random_state=0).fit(
        scaler.transform(values)
    )
    projected = pca.transform(scaler.transform(values))
    angle_scaler = StandardScaler().fit(projected)
    return scaler, pca, angle_scaler


def projected_angles(path, cache, projection):
    scaler, pca, angle_scaler = projection
    values = cache[_norm(path)][None, :]
    components = angle_scaler.transform(pca.transform(scaler.transform(values)))
    return np.tanh(components[0]) * np.pi


def entropy_pair_features(pairs, cache, projection):
    X, y = [], []
    entropy_cache = {}
    for first, second, label, relation in pairs:
        key_a, key_b = _norm(first), _norm(second)
        if key_a not in entropy_cache:
            entropy_cache[key_a] = state_entropy(
                projected_angles(first, cache, projection)
            )
        if key_b not in entropy_cache:
            entropy_cache[key_b] = state_entropy(
                projected_angles(second, cache, projection)
            )
        ea, eb = entropy_cache[key_a], entropy_cache[key_b]
        cosine = float(unit(cache[key_a]) @ unit(cache[key_b]))
        rel = pair_relation((first, second, label, relation))
        X.append([cosine, ea + eb, abs(ea - eb), *rel.tolist()])
        y.append(int(label))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def density_pair_features(pairs, cache, with_density):
    X, y = [], []
    for first, second, label, _relation in pairs:
        features = pair_set_features(
            cache[_norm(first)][None, :],
            cache[_norm(second)][None, :],
            with_density=with_density,
        )
        X.append(features_to_vector(features, with_density=with_density))
        y.append(int(label))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def logistic_result(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    classifier = LogisticRegression(max_iter=4000, C=1.0)
    classifier.fit(scaler.transform(X_train), y_train)
    scores = classifier.predict_proba(scaler.transform(X_test))[:, 1]
    return metric_row(y_test, scores)


def train_pca_entropy_fold(fold, cache):
    projection = fit_angle_projection(fold["train"], cache, 4)
    train_X, train_y = entropy_pair_features(fold["train"], cache, projection)
    test_X, test_y = entropy_pair_features(fold["test"], cache, projection)
    control = logistic_result(train_X[:, [0, 3, 4, 5, 6]], train_y,
                              test_X[:, [0, 3, 4, 5, 6]], test_y)
    treatment = logistic_result(train_X, train_y, test_X, test_y)
    return treatment, control


def amplitude_pair_states(pairs, cache, projection):
    states, labels = [], []
    for first, second, label, _relation in pairs:
        theta_a = projected_angles(first, cache, projection)[0:2]
        theta_b = projected_angles(second, cache, projection)[0:2]
        state_a = np.asarray(
            [math.cos(theta_a[0] / 2), math.sin(theta_a[0] / 2)]
        )
        state_b = np.asarray(
            [math.cos(theta_b[0] / 2), math.sin(theta_b[0] / 2)]
        )
        pair_state = np.kron(state_a, state_b) + np.kron(state_b, state_a)
        pair_state = pair_state / (np.linalg.norm(pair_state) + 1e-12)
        states.append(pair_state.astype(np.float32))
        labels.append(int(label))
    return np.stack(states), np.asarray(labels, dtype=np.int64)


class ProjectivePOVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.basis = nn.Parameter(torch.eye(4))

    def forward(self, states):
        basis, _ = torch.linalg.qr(self.basis)
        probabilities = (states @ basis).square()
        return probabilities[:, :2].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)


class MatchedLinearReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))

    def forward(self, states):
        probabilities = torch.softmax(states @ self.weight, dim=1)
        return probabilities[:, :2].sum(dim=1).clamp(1e-6, 1.0 - 1e-6)


def train_povm_model(model, train_states, train_y, test_states, epochs, seed,
                     device):
    torch.manual_seed(seed)
    model = model.to(device)
    device = next(model.parameters()).device
    x_train = torch.as_tensor(train_states, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(test_states, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = model(x_test).cpu().numpy()
    return scores


def train_povm_fold(fold, cache, args, seed):
    projection = fit_angle_projection(fold["train"], cache, 1)
    train_states, train_y = amplitude_pair_states(
        fold["train"], cache, projection
    )
    test_states, test_y = amplitude_pair_states(
        fold["test"], cache, projection
    )
    povm_scores = train_povm_model(
        ProjectivePOVM(), train_states, train_y, test_states,
        args.povm_epochs, seed, args.device,
    )
    linear_scores = train_povm_model(
        MatchedLinearReadout(), train_states, train_y, test_states,
        args.povm_epochs, seed, args.device,
    )
    return (
        metric_row(test_y, povm_scores),
        metric_row(test_y, linear_scores),
    )


def _phase_features(father, mother, child, alpha_points=21, phase_points=33):
    father, mother, child = unit(father), unit(mother), unit(child)
    alphas = np.linspace(0.0, 1.0, alpha_points)
    phases = np.linspace(0.0, 2.0 * np.pi, phase_points, endpoint=False)
    scores = np.empty((alpha_points, phase_points), dtype=np.float64)
    for ai, alpha in enumerate(alphas):
        for pi, phase in enumerate(phases):
            mixture = (
                alpha * np.exp(1j * phase) * father
                + (1.0 - alpha) * mother
            )
            mixture /= np.linalg.norm(mixture) + 1e-12
            scores[ai, pi] = abs(np.vdot(child, mixture)) ** 2
    classical = float(scores[:, 0].max())
    return np.asarray(
        [
            float(scores.max()),
            float(scores.mean()),
            float(scores.std()),
            float(scores.max() - classical),
        ],
        dtype=np.float64,
    )


def triad_rows(families, triads, cache, seed):
    keys = sorted(family for family in families if family in triads)
    rng = random.Random(seed)
    rows = []
    for family in keys:
        triad = triads[family]
        rows.append((triad["F"], triad["M"], triad["C"], 1, family))
    by_child_role = defaultdict(list)
    for family in keys:
        role = os.path.basename(triads[family]["C"]).rsplit(".", 1)[0].split("-")[-1]
        by_child_role[role].append(family)
    for donor_families in by_child_role.values():
        rng.shuffle(donor_families)
        if len(donor_families) < 2:
            continue
        for index, family in enumerate(donor_families):
            donor_family = donor_families[(index + 1) % len(donor_families)]
            rows.append((
                triads[family]["F"],
                triads[family]["M"],
                triads[donor_family]["C"],
                0,
                family,
            ))
    for father, mother, child, _label, _family in rows:
        if any(_norm(path) not in cache for path in (father, mother, child)):
            raise RuntimeError("Triadic sample lacks a cached embedding.")
    return rows


def triad_matrix(rows, cache, args, phase_cache):
    X, y = [], []
    for father, mother, child, label, _family in rows:
        ef, em, ec = (
            cache[_norm(father)], cache[_norm(mother)], cache[_norm(child)]
        )
        features = triad_features(ef, em, ec)
        vector = [features[key] for key in TRIAD_FEATURE_ORDER]
        key = (_norm(father), _norm(mother), _norm(child))
        if key not in phase_cache:
            phase_cache[key] = _phase_features(
                ef, em, ec, args.alpha_points, args.phase_points
            )
        vector.extend(phase_cache[key].tolist())
        X.append(vector)
        y.append(int(label))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def run_triad_fold(fold, triads, cache, args, seed, phase_cache):
    train_rows = triad_rows(
        fold["train_families"], triads, cache, seed + 1
    )
    test_rows = triad_rows(
        fold["test_families"], triads, cache, seed + 2
    )
    train_phase, train_y = triad_matrix(train_rows, cache, args, phase_cache)
    test_phase, test_y = triad_matrix(test_rows, cache, args, phase_cache)
    control = logistic_result(
        train_phase[:, :7], train_y, test_phase[:, :7], test_y
    )
    formulation4 = logistic_result(
        train_phase[:, [0, 1, 2, 3, 4, 5, 6, 10]],
        train_y,
        test_phase[:, [0, 1, 2, 3, 4, 5, 6, 10]],
        test_y,
    )
    formulation9 = logistic_result(train_phase, train_y, test_phase, test_y)
    test_families_a = [row[4] for row in test_rows]
    test_families_b = [family_of_ts(row[2]) for row in test_rows]
    for result in (control, formulation4, formulation9):
        if len(result["test_labels"]) != len(test_families_a):
            raise ValueError("Triad predictions do not match test families.")
        result["test_family_a"] = test_families_a
        result["test_family_b"] = test_families_b
    return control, formulation4, formulation9


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--purity-lambda", type=float, default=0.01)
    parser.add_argument("--povm-epochs", type=int, default=200)
    parser.add_argument("--alpha-points", type=int, default=21)
    parser.add_argument("--phase-points", type=int, default=33)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--tag", default="formulation_reruns_tskinface_reconstructed"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.folds != 5:
        parser.error("The reconstructed protocol requires five grouped folds.")
    if args.qubits < 4:
        parser.error("At least four qubits are needed for the purity readout.")
    output_path = os.path.join(OUTPUT_DIR, args.tag + ".json")
    if os.path.exists(output_path) and not args.overwrite:
        parser.error("Output exists; choose a new --tag or pass --overwrite.")
    if not os.path.isdir(TS_ROOT):
        raise FileNotFoundError("TSKinFace data not found: " + TS_ROOT)
    if not os.path.isfile(CACHE_PATH):
        raise FileNotFoundError("Embedding cache not found: " + CACHE_PATH)

    started = time.time()
    cache = load_cache()
    positive_pairs = load_positive_pairs(cache, args.seed)
    outer_folds = make_outer_folds(positive_pairs, args.seed, args.folds)
    triads = {
        family_of_ts(paths["F"]): paths
        for _family, paths in find_triads(TS_ROOT).items()
        if all(_norm(path) in cache for path in paths.values())
    }
    device = args.device
    print("TSKinFace: %d positive pair rows, %d families, cache coverage %.1f%%"
          % (
              len(positive_pairs),
              len({family_of_ts(pair[0]) for pair in positive_pairs}),
              100.0,
          ))
    print("device:", device, "| corrected grouped folds:", len(outer_folds))

    results = defaultdict(list)
    training = defaultdict(list)

    # Formulation 1: five seeded group-disjoint holdouts, quantum ON/OFF.
    for seed_index in range(5):
        seed = args.seed + seed_index
        train_positive, test_positive = list(
            grouped_kfold(positive_pairs, family_of_ts, 5, seed)
        )[0]
        fit, validation = make_inner_sides(train_positive, seed + 700)
        test = corrected_side(test_positive, seed + 3000)
        for use_quantum, name in (
            (True, "formulation_1_quantum"),
            (False, "formulation_1_control"),
        ):
            row, train_info = train_hybrid(
                fit, validation, test, cache, seed, args, device, use_quantum
            )
            add_pair_families(row, test)
            row["seed"] = seed
            results[name].append(row)
            training[name].append(train_info)
        print("Formulation 1 seed %d complete." % seed, flush=True)

    # Formulations 2, 7: matched five-fold pairwise-model runs.
    for fold in outer_folds:
        fit, validation = make_inner_sides(
            fold["train_positive"], args.seed + 4000 + fold["index"]
        )
        test = fold["test"]
        for use_quantum, name in (
            (True, "formulation_2_quantum"),
            (False, "formulation_2_control"),
        ):
            row, train_info = train_hybrid(
                fit, validation, test, cache,
                args.seed + 5000 + fold["index"], args, device, use_quantum
            )
            add_pair_families(row, test)
            row["fold"] = fold["index"] + 1
            results[name].append(row)
            training[name].append(train_info)
        for regularizer, name in (
            ("purity", "formulation_7_purity"),
            ("decorrelation", "formulation_7_control"),
        ):
            row, train_info = train_hybrid(
                fit, validation, test, cache,
                args.seed + 6000 + fold["index"], args, device,
                use_quantum=True, regularizer=regularizer,
            )
            add_pair_families(row, test)
            row["fold"] = fold["index"] + 1
            results[name].append(row)
            training[name].append(train_info)
        print("Formulations 2 and 7 fold %d complete."
              % (fold["index"] + 1), flush=True)

    # Formulations 3 and 8 share the same singleton-set density feature on TSKinFace.
    for fold in outer_folds:
        train_base, train_y = density_pair_features(
            fold["train"], cache, with_density=False
        )
        test_base, test_y = density_pair_features(
            fold["test"], cache, with_density=False
        )
        train_density, _ = density_pair_features(
            fold["train"], cache, with_density=True
        )
        test_density, _ = density_pair_features(
            fold["test"], cache, with_density=True
        )
        control = logistic_result(train_base, train_y, test_base, test_y)
        treatment = logistic_result(train_density, train_y, test_density, test_y)
        add_pair_families(control, fold["test"])
        add_pair_families(treatment, fold["test"])
        control["fold"] = treatment["fold"] = fold["index"] + 1
        results["formulation_3_control"].append(control)
        results["formulation_3_density"].append(treatment)
        results["formulation_8_control"].append(dict(control))
        results["formulation_8_density"].append(dict(treatment))
        print("Formulations 3 and 8 fold %d complete."
              % (fold["index"] + 1), flush=True)

    # Formulation 5: four-qubit entropy feature added to cosine and relation.
    for fold in outer_folds:
        treatment, control = train_pca_entropy_fold(fold, cache)
        add_pair_families(treatment, fold["test"])
        add_pair_families(control, fold["test"])
        treatment["fold"] = control["fold"] = fold["index"] + 1
        results["formulation_5_entropy"].append(treatment)
        results["formulation_5_control"].append(control)
        print("Formulation 5 fold %d complete."
              % (fold["index"] + 1), flush=True)

    # Formulation 6: four-outcome projective POVM versus a 16-parameter head.
    for fold in outer_folds:
        treatment, control = train_povm_fold(
            fold, cache, args, args.seed + 8000 + fold["index"]
        )
        add_pair_families(treatment, fold["test"])
        add_pair_families(control, fold["test"])
        treatment["fold"] = control["fold"] = fold["index"] + 1
        results["formulation_6_povm"].append(treatment)
        results["formulation_6_control"].append(control)
        print("Formulation 6 fold %d complete."
              % (fold["index"] + 1), flush=True)

    # Formulations 4 and 9: construct negatives only within each outer side.
    phase_cache = {}
    for fold in outer_folds:
        control, formulation4, formulation9 = run_triad_fold(
            fold, triads, cache, args, args.seed + 9000 + fold["index"],
            phase_cache,
        )
        control["fold"] = formulation4["fold"] = formulation9["fold"] = (
            fold["index"] + 1
        )
        results["formulation_4_control"].append(dict(control))
        results["formulation_4_phase"].append(formulation4)
        results["formulation_9_control"].append(dict(control))
        results["formulation_9_phase"].append(formulation9)
        print("Formulations 4 and 9 fold %d complete."
              % (fold["index"] + 1), flush=True)

    comparisons = {}
    comparison_specs = (
        ("formulation_2", "formulation_2_quantum", "formulation_2_control"),
        ("formulation_3", "formulation_3_density", "formulation_3_control"),
        ("formulation_4", "formulation_4_phase", "formulation_4_control"),
        ("formulation_5", "formulation_5_entropy", "formulation_5_control"),
        ("formulation_6", "formulation_6_povm", "formulation_6_control"),
        ("formulation_7", "formulation_7_purity", "formulation_7_control"),
        ("formulation_9", "formulation_9_phase", "formulation_9_control"),
    )
    for name, treatment, control in comparison_specs:
        comparisons[name] = paired_comparison(
            results[treatment], results[control]
        )
        comparisons[name]["family_cluster_bootstrap"] = cluster_bootstrap_delta(
            results[treatment],
            results[control],
            seed=args.seed + 10000 + len(comparisons),
            replicates=args.bootstrap_replicates,
        )
    comparisons["formulation_8"] = {
        "alias_of": "formulation_3",
        "interpretation": (
            "Same corrected TSKinFace singleton-set density experiment; "
            "not an independent comparison."
        ),
    }
    f1_deltas = np.asarray([
        on["roc_auc"] - off["roc_auc"]
        for on, off in zip(
            results["formulation_1_quantum"],
            results["formulation_1_control"],
        )
    ])
    comparisons["formulation_1"] = {
        "delta_mean": float(f1_deltas.mean()),
        "fold_deltas": f1_deltas.tolist(),
        "interpretation": (
            "Descriptive five-seed holdout differences; no inferential test "
            "is reported."
        ),
    }

    summary = {name: summarize_rows(rows) for name, rows in results.items()}
    config = {
        "run_type": "reconstructed, corrected TSKinFace-only formulation rerun",
        "source_commit": git_commit(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        ),
        "cli_args": {**vars(args), "device": str(args.device)},
        "positive_pair_rows": len(positive_pairs),
        "positive_families": len({
            family_of_ts(pair[0]) for pair in positive_pairs
        }),
        "embedding_cache_entries": len(cache),
        "embedding_coverage": "100% for every pair endpoint",
        "pair_protocol": {
            "outer_split": "five family-grouped folds",
            "negative_construction": (
                "after splitting; 50% father-vs-mother within-family "
                "negatives; remaining negatives cross-family but remain "
                "within the same split"
            ),
            "triadic_negatives": (
                "replace each family's child with a child from another "
                "family on the same side of the outer split"
            ),
        },
        "reconstruction_assumptions": {
            "formulation_1": (
                "five seeded 80/20 group holdouts on TSKinFace, adapting the "
                "historical FIW five-seed protocol"
            ),
            "formulation_2": (
                "five matched TSKinFace folds, not the historical 20 folds "
                "across four corpora"
            ),
            "formulations_3_and_8": (
                "same singleton-set density feature and control on TSKinFace; "
                "reported as one operationalization, not independent evidence"
            ),
            "formulation_4": (
                "amplitude-encoded normalized embeddings, alpha sweep over "
                "[0,1], relative phase sweep over [0,2pi), phase-gain feature"
            ),
            "formulation_5": (
                "training-fold-fitted PCA and standardization to four Ry "
                "angles; fixed Ry/CNOT-chain state; entropy of a two-qubit "
                "reduced density matrix; local Rz gates omitted because they "
                "do not change bipartite entropy"
            ),
            "formulation_6": (
                "two-qubit symmetric amplitude state; four rank-one "
                "orthogonal POVM effects; outcomes 0 and 1 map to kin; "
                "16-parameter four-logit linear control"
            ),
            "formulation_7": (
                "current symmetric hybrid classifier; four-qubit reduced "
                "state purity penalty versus batch decorrelation penalty; "
                "both weighted by lambda=0.01"
            ),
            "formulation_9": (
                "triad features augmented by maximum, mean, standard "
                "deviation, and gain over the zero-phase amplitude-encoded "
                "interference sweep"
            ),
        },
        "statistical_caveat": (
            "Five folds partition the same family pool. Paired t tests and "
            "fold-based intervals are exploratory. Family-cluster bootstrap "
            "intervals use fixed out-of-fold predictions and condition on the "
            "fitted models; they do not capture training-set variability. "
            "Multiplicity adjustment does not make folds independent "
            "population samples."
        ),
        "elapsed_seconds": time.time() - started,
    }
    output = {
        "config": config,
        "results": summary,
        "paired_comparisons": comparisons,
        "training": dict(training),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temporary_path = output_path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    print("Saved:", os.path.relpath(output_path, PROJECT_ROOT))
    for name, result in summary.items():
        if name.endswith("_control"):
            continue
        print("%-28s AUC %.4f +/- %.4f, accuracy %.2f%%"
              % (
                  name,
                  result["roc_auc_mean"],
                  result["roc_auc_sd"],
                  100 * result["accuracy_mean"],
              ))


if __name__ == "__main__":
    main()
