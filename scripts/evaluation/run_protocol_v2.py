# -*- coding: utf-8 -*-
"""Protocol v2 reruns. See docs/PROTOCOL_V2.md for the analysis plan.

Each run appends to results/protocol_v2/<experiment>.jsonl and is skipped on
restart, so a long job can stop and resume. Scores go to scores/*.npz.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import load_fiw_pairs, load_kinfacew_pairs
from src.fair_protocol import (ColourDistance, build_side, drop_duplicate_groups,
                               is_failed_run, split_fit_val)
from src.identity_sets import features_to_vector, identity_of_path, pair_set_features
from src.kfold import balance_group_sizes, grouped_kfold
from src.models_fair import FairKinshipModel, RelationFreeHybrid, ScalarReadoutModel
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.models_vqc import (AmplitudeControl, AmplitudeVQCClassifier,
                            WidthMatchedControl)
from src.multi_dataset import _kfw_family
from src.rfiw import PARENT_CHILD, load_identity_pairs, load_labelled_pairs
from src.splits import family_of
from src.triads import TRIAD_FEATURE_ORDER, find_triads, triad_features
from src.ts_pairs import add_tskinface_negatives, build_tskinface_pairs, family_of_ts

OUT = os.path.join(ROOT, "results", "protocol_v2")
CACHES = {"facenet": "all_datasets_cache.pkl", "arcface": "arcface_cache.pkl"}
TS_ROOT = os.path.join(ROOT, "TSKinFace_Data", "TSKinFace_Data", "TSKinFace_cropped")
SPLIT_SEED = 42
FOLDS = 5
SEEDS = (0, 1, 2)
NEGATIVE_SEED_OFFSET = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLOUR = ColourDistance()


# ---------------------------------------------------------------- bookkeeping
def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   stderr=subprocess.DEVNULL).strip()


def source_fingerprint():
    """Identify the exact local Python sources used by this protocol run."""
    relative_paths = (
        "scripts/evaluation/run_protocol_v2.py",
        "scripts/evaluation/rerun_formulations_tskinface.py",
        "src/calibration.py",
        "src/data_loaders.py",
        "src/fair_protocol.py",
        "src/identity_sets.py",
        "src/kfold.py",
        "src/models_fair.py",
        "src/models_hybrid.py",
        "src/models_vqc.py",
        "src/multi_dataset.py",
        "src/quantum_core.py",
        "src/quantum_fast.py",
        "src/quantum_vqc.py",
        "src/rfiw.py",
        "src/splits.py",
        "src/triads.py",
        "src/ts_pairs.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = os.path.join(ROOT, *relative.split("/"))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as stream:
            digest.update(stream.read())
        digest.update(b"\0")
    return {"source_sha256": digest.hexdigest(), "source_file_count": len(relative_paths)}


PROVENANCE = {
    "commit": _git("rev-parse", "HEAD"),
    "dirty_tree": bool(_git("status", "--porcelain")),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    **source_fingerprint(),
}


class Log:
    """Append-only JSONL. A run key already present is not repeated."""

    def __init__(self, name):
        os.makedirs(os.path.join(OUT, "scores"), exist_ok=True)
        self.path = os.path.join(OUT, name + ".jsonl")
        self.done = set()
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("negative_seed_offset") != NEGATIVE_SEED_OFFSET:
                        raise ValueError(
                            f"{self.path} contains a different or unrecorded negative-seed "
                            "offset; use a fresh --output-dir")
                    provenance = record.get("provenance", {})
                    if provenance.get("source_sha256") != PROVENANCE["source_sha256"]:
                        raise ValueError(
                            f"{self.path} was produced by different protocol source; "
                            "use a fresh --output-dir")
                    self.done.add(record["key"])

    def write(self, key, **record):
        record = {"key": key, **record, "negative_seed_offset": NEGATIVE_SEED_OFFSET,
                  "provenance": PROVENANCE}
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.done.add(key)
        print(key, {k: round(v, 4) for k, v in record.items()
                    if isinstance(v, float)}, flush=True)


def save_scores(name, **arrays):
    np.savez_compressed(os.path.join(OUT, "scores", name + ".npz"), **arrays)


# ----------------------------------------------------------------------- data
def norm(path):
    return os.path.normcase(os.path.abspath(path))


def pair_groups(pairs, keyfn):
    return (
        np.asarray([keyfn(pair[0]) for pair in pairs]),
        np.asarray([keyfn(pair[1]) for pair in pairs]),
    )


def negative_seed(seed):
    """Vary negative sampling without changing folds or model initialization."""
    return int(seed) + NEGATIVE_SEED_OFFSET


def load_cache(file_name):
    with open(os.path.join(ROOT, "weights", "caches", file_name), "rb") as stream:
        return {norm(k): np.asarray(v, dtype=np.float32)
                for k, v in pickle.load(stream).items()}


def covered(pairs, cache):
    return [p for p in pairs if norm(p[0]) in cache and norm(p[1]) in cache]


def key_for(corpus):
    if corpus == "FIW":
        return family_of
    if corpus == "TSKinFace":
        return family_of_ts
    return _kfw_family


def load_positives(cache, cap):
    os.chdir(ROOT)
    fiw = [p for p in load_fiw_pairs("public") if p[2] == 1]
    out = {"FIW": balance_group_sizes(fiw, family_of, cap, SPLIT_SEED)}
    for folder in ("KinFaceW-I", "KinFaceW-II"):
        out[folder] = [r[:4] for r in load_kinfacew_pairs(folder) if r[2] == 1]
    out["TSKinFace"] = [p for p in build_tskinface_pairs(TS_ROOT, 0.5, SPLIT_SEED)
                        if p[2] == 1]
    result = {}
    for corpus, pairs in out.items():
        pairs = covered(pairs, cache)
        pairs, dropped = drop_duplicate_groups(
            pairs, key_for(corpus), lambda path: cache[norm(path)])
        if dropped:
            print("  %s: dropped duplicate families %s" % (corpus, dropped), flush=True)
        result[corpus] = pairs
    return result


def old_negatives(positives, keyfn, seed):
    """The pre-v2 sampler, kept only to measure how much it inflated scores."""
    if keyfn is family_of_ts:
        return add_tskinface_negatives(positives, keyfn=keyfn, seed=seed,
                                       same_family_negative_ratio=0.5)
    rng = random.Random(seed)
    pool = {}
    for a, b, _l, _r in positives:
        pool.setdefault(keyfn(a), []).append(a)
        pool.setdefault(keyfn(b), []).append(b)
    keys = sorted(pool)
    negatives = []
    for a, _b, _l, relation in positives:
        others = [k for k in keys if k != keyfn(a)]
        choice = pool[others[rng.randrange(len(others))]]
        negatives.append((a, choice[rng.randrange(len(choice))], 0, relation))
    return list(positives) + negatives


REL_INDEX = {"fs": 0, "fd": 1, "ms": 2, "md": 3}


def tensors(pairs, cache):
    a = np.stack([cache[norm(p[0])] for p in pairs])
    b = np.stack([cache[norm(p[1])] for p in pairs])
    y = np.array([float(p[2]) for p in pairs], dtype=np.float32)
    r = np.zeros((len(pairs), 4), dtype=np.float32)
    for i, pair in enumerate(pairs):
        r[i, REL_INDEX.get(pair[3], 0)] = 1.0
    return tuple(torch.as_tensor(x, device=DEVICE) for x in (a, b, y, r))


def colour_scores(pairs):
    """Similarity from image colour alone: no face identity is involved."""
    return np.array([-COLOUR(p[0], p[1]) for p in pairs])


def cosine_scores(data):
    return F.cosine_similarity(data[0], data[1], dim=1).cpu().numpy()


def auc(data, scores):
    return float(roc_auc_score(data[2].cpu().numpy(), scores))


def cap_pairs(pairs, limit, seed):
    if len(pairs) <= limit:
        return pairs
    rng = np.random.default_rng(seed)
    return [pairs[i] for i in rng.choice(len(pairs), limit, replace=False)]


# ------------------------------------------------------------------- training
def fit_model(model, score_fn, loss_fn, fit, val, epochs, lr, weight_decay,
              batch_size, patience, seed):
    """Early-stopped training on validation AUC. `loss_fn(model, batch)`."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, since, ran = -1.0, None, 0, 0
    for _ in range(epochs):
        ran += 1
        model.train()
        order = torch.randperm(len(fit[2]), device=DEVICE)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            optimizer.zero_grad()
            loss = loss_fn(model, tuple(t[index] for t in fit))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        value = auc(val, score_fn(model, val))
        if value > best:
            best, since = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return best, ran


@torch.no_grad()
def batched_scores(model, data, call, size=2048):
    model.eval()
    return torch.cat([call(model, *(t[i:i + size] for t in data)).view(-1).cpu()
                      for i in range(0, len(data[2]), size)]).numpy()


def fair_call(model, a, b, _y, _r):
    return model(a, b)


def hybrid_call(zero_relation):
    def call(model, a, b, _y, r):
        r = torch.zeros_like(r) if zero_relation else r
        return 0.5 * (model(a, b, r) + model(b, a, r))
    return call


def predict_call(model, a, b, _y, r):
    return model.predict(a, b, r)


def build_arm(arm):
    """(model, scorer, batched loss, lr, weight decay, batch size, epochs)."""
    bce = nn.BCEWithLogitsLoss()
    if arm == "fair":
        model = FairKinshipModel()
        loss = lambda m, b: bce(m.forward_logits(b[0], b[1]).view(-1), b[2])
        return model, fair_call, loss, 3e-3, 1e-2, 512, 40
    if arm.startswith("hybrid"):
        # hybrid_paper_*: the paper's model, relation input on.
        # hybrid_q0/q1: the same head without the relation path.
        zero = not arm.startswith("hybrid_paper")
        family = RelationFreeHybrid if zero else QuantumAugmentedKinshipClassifier
        model = family(dropout=0.5, use_quantum=not arm.endswith("q0"))

        def loss(m, b):
            r = torch.zeros_like(b[3]) if zero else b[3]
            return bce(m.forward_logits(b[0], b[1], r).view(-1), b[2])
        return model, hybrid_call(zero), loss, 1e-3, 1e-2, 512, 40
    raise ValueError(arm)


def lr_features(data):
    a = F.normalize(data[0], dim=1).cpu().numpy()
    b = F.normalize(data[1], dim=1).cpu().numpy()
    return np.c_[np.abs(a - b), a * b, (a * b).sum(1, keepdims=True)]


def threshold_row(val, val_scores, test, test_scores):
    threshold = calibrate_threshold(val[2].cpu().numpy(), val_scores, "accuracy",
                                    max_fpr=0.25, smooth=True)
    point = operating_point(test[2].cpu().numpy(), test_scores, threshold)
    return {"accuracy": point["accuracy"], "fpr": point["fpr"],
            "threshold": float(threshold)}


# --------------------------------------------------------- reference (Q1, Q2)
def run_reference(backbone, seeds):
    log = Log("reference_" + backbone)
    cache = load_cache(CACHES[backbone])
    positives = load_positives(cache, 1500)
    folds = {c: list(grouped_kfold(p, key_for(c), FOLDS, SPLIT_SEED))
             for c, p in positives.items()}
    for fold in range(FOLDS):
        sides = {}
        fit_new, val_new, fit_old, val_old, fit_col, val_col = [], [], [], [], [], []
        for corpus, corpus_folds in folds.items():
            train, test = corpus_folds[fold]
            keyfn = key_for(corpus)
            fit_pos, val_pos = split_fit_val(train, keyfn, SPLIT_SEED + fold)
            fit_new += cap_pairs(covered(build_side(
                fit_pos, keyfn, negative_seed(SPLIT_SEED + fold)), cache),
                8000, negative_seed(SPLIT_SEED + fold))
            val_part = covered(build_side(
                val_pos, keyfn, negative_seed(SPLIT_SEED + 100 + fold)), cache)
            val_new += val_part
            fit_old += cap_pairs(covered(old_negatives(
                fit_pos, keyfn, negative_seed(SPLIT_SEED + fold)), cache),
                8000, negative_seed(SPLIT_SEED + fold))
            val_old += covered(old_negatives(
                val_pos, keyfn, negative_seed(SPLIT_SEED + 100 + fold)), cache)
            fit_col += cap_pairs(covered(build_side(
                fit_pos, keyfn, negative_seed(SPLIT_SEED + fold), COLOUR), cache),
                8000, negative_seed(SPLIT_SEED + fold))
            val_col += covered(build_side(
                val_pos, keyfn, negative_seed(SPLIT_SEED + 100 + fold), COLOUR), cache)
            sides[corpus] = {
                "matched": covered(build_side(
                    test, keyfn, negative_seed(SPLIT_SEED + 500 + fold)), cache),
                "colour_matched": covered(build_side(
                    test, keyfn, negative_seed(SPLIT_SEED + 500 + fold), COLOUR), cache),
                "old": covered(old_negatives(
                    test, keyfn, negative_seed(SPLIT_SEED + 500 + fold)), cache),
                "val": val_part,
            }
        test_data = {c: {k: tensors(v, cache) for k, v in s.items()}
                     for c, s in sides.items()}
        Fn, Vn = tensors(fit_new, cache), tensors(val_new, cache)
        Fo, Vo = tensors(fit_old, cache), tensors(val_old, cache)
        Fc, Vc = tensors(fit_col, cache), tensors(val_col, cache)

        def record(arm, seed, scores_by, extra):
            for corpus, per in scores_by.items():
                data = test_data[corpus]
                row = {"auc_matched": auc(data["matched"], per["matched"]),
                       "auc_old": auc(data["old"], per["old"]),
                       "auc_colour_matched": auc(data["colour_matched"], per["colour_matched"]),
                       **threshold_row(data["val"], per["val"],
                                       data["matched"], per["matched"])}
                log.write(f"{arm}|{corpus}|f{fold}|s{seed}", arm=arm, corpus=corpus,
                          fold=fold, seed=seed, **row, **extra)
                save_scores(f"reference_{backbone}_{arm}_{corpus}_f{fold}_s{seed}",
                            matched=per["matched"].astype(np.float32),
                            y=data["matched"][2].cpu().numpy().astype(np.int8),
                            groups_a=pair_groups(
                                sides[corpus]["matched"], key_for(corpus)
                            )[0],
                            groups_b=pair_groups(
                                sides[corpus]["matched"], key_for(corpus)
                            )[1],
                            colour_matched=per["colour_matched"].astype(np.float32),
                            y_colour=data["colour_matched"][2].cpu().numpy().astype(np.int8),
                            groups_colour_a=pair_groups(
                                sides[corpus]["colour_matched"], key_for(corpus)
                            )[0],
                            groups_colour_b=pair_groups(
                                sides[corpus]["colour_matched"], key_for(corpus)
                            )[1])

        if f"cosine|TSKinFace|f{fold}|s0" not in log.done:
            record("cosine", 0, {c: {k: cosine_scores(d) for k, d in per.items()}
                                 for c, per in test_data.items()},
                   {"failed": False, "val_auc": auc(Vn, cosine_scores(Vn))})
        if f"colour|TSKinFace|f{fold}|s0" not in log.done:
            record("colour", 0, {c: {k: colour_scores(sides[c][k]) for k in per}
                                 for c, per in test_data.items()},
                   {"failed": False})
        if f"lr|TSKinFace|f{fold}|s0" not in log.done:
            scaler = StandardScaler().fit(lr_features(Fn))
            clf = LogisticRegression(max_iter=500, C=1e-3).fit(
                scaler.transform(lr_features(Fn)), Fn[2].cpu().numpy())
            score = lambda d: clf.decision_function(scaler.transform(lr_features(d)))
            record("lr", 0, {c: {k: score(d) for k, d in per.items()}
                             for c, per in test_data.items()},
                   {"failed": False, "val_auc": auc(Vn, score(Vn))})

        for arm, fit, val in (("fair", Fn, Vn), ("fair_colour_trained", Fc, Vc),
                              ("hybrid_q0", Fn, Vn), ("hybrid_q1", Fn, Vn),
                              ("hybrid_paper_q1", Fo, Vo)):
            cosine_val = auc(val, cosine_scores(val))
            for seed in seeds:
                if f"{arm}|TSKinFace|f{fold}|s{seed}" in log.done:
                    continue
                torch.manual_seed(seed + 1000 * fold)
                np.random.seed(seed + 1000 * fold)
                model, call, loss, lr, wd, bs, epochs = build_arm(arm.replace("_colour_trained", ""))
                model.to(DEVICE)
                started = time.time()
                score = lambda m, d: batched_scores(m, d, call)
                best, ran = fit_model(model, score, loss, fit, val, epochs, lr, wd, bs, 8, seed)
                train_auc = auc(fit, score(model, fit))
                train_cosine_auc = auc(fit, cosine_scores(fit))
                record(arm, seed, {c: {k: score(model, d) for k, d in per.items()}
                                   for c, per in test_data.items()},
                       {"val_auc": best, "train_auc": train_auc,
                        "train_cosine_auc": train_cosine_auc,
                        "epochs": ran, "seconds": time.time() - started,
                        "val_cosine_auc": cosine_val,
                        "failed": bool(is_failed_run(best, cosine_val))})
        del Fn, Vn, Fo, Vo, Fc, Vc, test_data
        torch.cuda.empty_cache()


# -------------------------------------------------------------- readout (Q3, Q4)
READOUT_ARMS = ("expectation", "fidelity", "fidelity_log", "swap", "control",
                "width_linear", "width_mlp", "width_random")


def build_readout(arm):
    if arm in ("expectation", "fidelity"):
        return AmplitudeVQCClassifier(n_qubits_each=6, depth=4, dropout=0.3, readout=arm)
    if arm in ("fidelity_log", "swap"):
        return ScalarReadoutModel(kind=arm, n_qubits_each=6, depth=4, dropout=0.3)
    if arm == "control":
        return AmplitudeControl(n_qubits_each=6, depth=4, dropout=0.3)
    return WidthMatchedControl(n_qubits_each=6, depth=4, dropout=0.3,
                               mode=arm.split("_")[1])


def run_readout(backbone, seeds):
    log = Log("readout_" + backbone)
    cache = load_cache(CACHES[backbone])
    positives = load_positives(cache, 200)
    bce = nn.BCEWithLogitsLoss()
    for corpus, pairs in positives.items():
        keyfn = key_for(corpus)
        for fold, (train, test) in enumerate(grouped_kfold(pairs, keyfn, FOLDS, SPLIT_SEED)):
            fit_pos, val_pos = split_fit_val(train, keyfn, SPLIT_SEED + fold)
            Fit = tensors(covered(build_side(
                fit_pos, keyfn, negative_seed(SPLIT_SEED + fold)), cache), cache)
            Val = tensors(covered(build_side(
                val_pos, keyfn, negative_seed(SPLIT_SEED + 100 + fold)), cache), cache)
            test_pairs = covered(build_side(
                test, keyfn, negative_seed(SPLIT_SEED + 500 + fold)), cache)
            Test = tensors(test_pairs, cache)
            colour_pairs = covered(build_side(
                test, keyfn, negative_seed(SPLIT_SEED + 500 + fold), COLOUR), cache)
            TestC = tensors(colour_pairs, cache)
            cosine = auc(Test, cosine_scores(Test))
            cosine_c = auc(TestC, cosine_scores(TestC))
            colour_auc = auc(Test, colour_scores(test_pairs))
            for arm in READOUT_ARMS:
                for seed in seeds:
                    key = f"{arm}|{corpus}|f{fold}|s{seed}"
                    if key in log.done:
                        continue
                    torch.manual_seed(seed + 1000 * fold)
                    np.random.seed(seed + 1000 * fold)
                    model = build_readout(arm).to(DEVICE)

                    def loss(m, b):
                        return bce(m.forward_logits(b[0], b[1], b[3]).view(-1), b[2])
                    score = lambda m, d: batched_scores(m, d, predict_call, 1024)
                    started = time.time()
                    best, ran = fit_model(model, score, loss, Fit, Val, 25, 3e-3, 1e-3,
                                          256, 8, seed)
                    test_scores = score(model, Test)
                    val_scores = score(model, Val)
                    colour_view = score(model, TestC)
                    log.write(key, arm=arm, corpus=corpus, fold=fold, seed=seed,
                              auc_matched=auc(Test, test_scores),
                              auc_colour_matched=auc(TestC, colour_view), val_auc=best,
                              cosine=cosine, cosine_colour_matched=cosine_c,
                              colour_only=colour_auc, epochs=ran,
                              seconds=time.time() - started,
                              **threshold_row(Val, val_scores, Test, test_scores))
                    groups_a, groups_b = pair_groups(test_pairs, keyfn)
                    colour_groups_a, colour_groups_b = pair_groups(colour_pairs, keyfn)
                    save_scores(f"readout_{backbone}_{arm}_{corpus}_f{fold}_s{seed}",
                                        matched=test_scores.astype(np.float32),
                                        y=Test[2].cpu().numpy().astype(np.int8),
                                        groups_a=groups_a, groups_b=groups_b,
                                        colour_matched=colour_view.astype(np.float32),
                                        y_colour=TestC[2].cpu().numpy().astype(np.int8),
                                        groups_colour_a=colour_groups_a,
                                        groups_colour_b=colour_groups_b)
            del Fit, Val, Test, TestC
            torch.cuda.empty_cache()


# ------------------------------------------------------------ set-level (Q5, Q6)
def source_photo(path):
    return os.path.basename(path).split("_")[0]


def shared_source_photos(paths_a, paths_b):
    return {source_photo(path) for path in paths_a} & {
        source_photo(path) for path in paths_b
    }


def galleries(pairs, cache):
    out = defaultdict(set)
    for first, second, _l, _r in pairs:
        for path in (first, second):
            identity = identity_of_path(path)
            candidates = [path]
            if os.sep + "fids" + os.sep in os.path.normcase(os.path.abspath(path)):
                person_dir = os.path.dirname(path)
                candidates = []
                for pattern in ("*.jpg", "*.jpeg", "*.png"):
                    candidates.extend(glob.glob(os.path.join(person_dir, pattern)))
            out[identity].update(p for p in candidates if norm(p) in cache)
    return out


def set_matrix(pairs, gallery, cache, mode):
    """Features per photo pair. Identity-level results are memoised per pair."""
    memo = {}
    rows, labels, kept = [], [], []
    density = mode == "set+density"
    for index, (first, second, label, _r) in enumerate(pairs):
        if mode in ("single", "single_noshared"):
            if mode == "single_noshared":
                A = gallery[identity_of_path(first)]
                B = gallery[identity_of_path(second)]
                shared = shared_source_photos(A, B)
                if source_photo(first) in shared or source_photo(second) in shared:
                    continue
            features = pair_set_features(cache[norm(first)][None], cache[norm(second)][None])
            rows.append(features_to_vector(features))
            labels.append(label)
            kept.append(index)
            continue
        key = (identity_of_path(first), identity_of_path(second))
        if key not in memo:
            A, B = sorted(gallery[key[0]]), sorted(gallery[key[1]])
            if mode == "set_noshared":
                shared = shared_source_photos(A, B)
                if source_photo(first) in shared or source_photo(second) in shared:
                    continue
                A = [p for p in A if source_photo(p) not in shared]
                B = [p for p in B if source_photo(p) not in shared]
            memo[key] = None
            if A and B:
                features = pair_set_features(
                    np.stack([cache[norm(p)] for p in A]),
                    np.stack([cache[norm(p)] for p in B]), with_density=density)
                memo[key] = features_to_vector(features, with_density=density)
        if memo[key] is not None:
            rows.append(memo[key])
            labels.append(label)
            kept.append(index)
    if not rows:
        raise ValueError(f"No usable identity pairs for set mode {mode!r}")
    return np.stack(rows), np.array(labels), np.array(kept)


def run_setlevel(backbone):
    log = Log("setlevel_" + backbone)
    cache = load_cache(CACHES[backbone])
    positives = load_positives(cache, 1500)
    for corpus, pairs in positives.items():
        keyfn = key_for(corpus)
        for fold, (train, test) in enumerate(grouped_kfold(pairs, keyfn, FOLDS, SPLIT_SEED)):
            tr = covered(build_side(train, keyfn, negative_seed(SPLIT_SEED + fold)), cache)
            te = covered(build_side(test, keyfn, negative_seed(SPLIT_SEED + 500 + fold)), cache)
            g_tr, g_te = galleries(tr, cache), galleries(te, cache)
            for arm in ("single", "set", "set+density", "set_noshared",
                        "single_noshared"):
                key = f"{arm}|{corpus}|f{fold}|s0"
                if key in log.done or (arm.endswith("_noshared") and corpus != "FIW"):
                    continue
                Xtr, ytr, _ = set_matrix(tr, g_tr, cache, arm)
                Xte, yte, kept = set_matrix(te, g_te, cache, arm)
                scaler = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=4000).fit(scaler.transform(Xtr), ytr)
                scores = clf.decision_function(scaler.transform(Xte))
                log.write(key, arm=arm, corpus=corpus, fold=fold, seed=0,
                          auc_matched=float(roc_auc_score(yte, scores)),
                          n=int(len(yte)), positive_rate=float(yte.mean()))
                save_scores(f"setlevel_{backbone}_{arm}_{corpus}_f{fold}_s0",
                            matched=scores.astype(np.float32), y=yte.astype(np.int8),
                            index=kept.astype(np.int32),
                            groups_a=np.array([keyfn(te[i][0]) for i in kept]),
                            groups_b=np.array([keyfn(te[i][1]) for i in kept]))


# ---------------------------------------------------------------- triadic (Q7)
def load_reconstructed():
    path = os.path.join(ROOT, "scripts", "evaluation", "rerun_formulations_tskinface.py")
    spec = importlib.util.spec_from_file_location("rerun_formulations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_triadic(backbone):
    log = Log("triadic_" + backbone)
    recon = load_reconstructed()
    cache = load_cache(CACHES[backbone])
    positives = load_positives(cache, 1500)["TSKinFace"]
    triads = {family_of_ts(paths["F"]): paths for paths in find_triads(TS_ROOT).values()
              if all(norm(p) in cache for p in paths.values())}
    phase_memo = {}

    groups_of = lambda rows: (
        np.array([row[4] for row in rows]),
        np.array([family_of_ts(row[2]) for row in rows]),
    )

    def matrix(rows):
        X, y = [], []
        for father, mother, child, label, _family in rows:
            ef, em, ec = (cache[norm(p)] for p in (father, mother, child))
            features = triad_features(ef, em, ec)
            key = (norm(father), norm(mother), norm(child))
            if key not in phase_memo:
                phase_memo[key] = recon._phase_features(ef, em, ec, 21, 33)
            X.append([features[k] for k in TRIAD_FEATURE_ORDER] + phase_memo[key].tolist()
                     + [max(features["cos_fc"], features["cos_mc"])])
            y.append(label)
        return np.asarray(X, dtype=np.float64), np.asarray(y)

    n_triad = len(TRIAD_FEATURE_ORDER)
    arms = {"best_parent": [n_triad + 4],
            "triad": list(range(n_triad)),
            "triad+phase_gain": list(range(n_triad)) + [n_triad + 3],
            "triad+phase_all": list(range(n_triad + 4))}
    def colour_rows(families, seed):
        """Same rows, but each donor child is the closest colour-distance match."""
        rng = random.Random(seed)
        keys = sorted(f for f in families if f in triads)
        rows = []
        role = lambda f: os.path.basename(triads[f]["C"]).rsplit(".", 1)[0].split("-")[-1]
        spread = lambda t, child: (COLOUR(t["F"], child) + COLOUR(t["M"], child)) / 2.0
        for family in keys:
            t = triads[family]
            rows.append((t["F"], t["M"], t["C"], 1, family))
            donors = [f for f in keys if f != family and role(f) == role(family)]
            sample = rng.sample(donors, min(32, len(donors)))
            target = spread(t, t["C"])
            best = min(sample, key=lambda f: abs(spread(t, triads[f]["C"]) - target))
            rows.append((t["F"], t["M"], triads[best]["C"], 0, family))
        return rows

    for fold, (train, test) in enumerate(grouped_kfold(positives, family_of_ts, FOLDS, SPLIT_SEED)):
        tr_f = {family_of_ts(p[0]) for p in train}
        te_f = {family_of_ts(p[0]) for p in test}
        Xtr, ytr = matrix(recon.triad_rows(
            tr_f, triads, cache, negative_seed(SPLIT_SEED + fold)))
        test_rows = recon.triad_rows(
            te_f, triads, cache, negative_seed(SPLIT_SEED + 500 + fold))
        colour_test_rows = colour_rows(
            te_f, negative_seed(SPLIT_SEED + 500 + fold))
        Xte, yte = matrix(test_rows)
        Xtc, ytc = matrix(colour_test_rows)
        for arm, columns in arms.items():
            key = f"{arm}|TSKinFace|f{fold}|s0"
            if key in log.done:
                continue
            scaler = StandardScaler().fit(Xtr[:, columns])
            clf = LogisticRegression(max_iter=4000).fit(scaler.transform(Xtr[:, columns]), ytr)
            scores = clf.decision_function(scaler.transform(Xte[:, columns]))
            colour_view = clf.decision_function(scaler.transform(Xtc[:, columns]))
            log.write(key, arm=arm, corpus="TSKinFace", fold=fold, seed=0,
                      auc_matched=float(roc_auc_score(yte, scores)),
                      auc_colour_matched=float(roc_auc_score(ytc, colour_view)), n=int(len(yte)))
            test_groups_a, test_groups_b = groups_of(test_rows)
            colour_groups_a, colour_groups_b = groups_of(colour_test_rows)
            save_scores(f"triadic_{backbone}_{arm}_TSKinFace_f{fold}_s0",
                            matched=scores.astype(np.float32), y=yte.astype(np.int8),
                            groups_a=test_groups_a, groups_b=test_groups_b,
                            colour_matched=colour_view.astype(np.float32),
                            y_colour=ytc.astype(np.int8),
                            groups_colour_a=colour_groups_a, groups_colour_b=colour_groups_b)


# -------------------------------------------------------------- formulations (Q5)
def run_formulations(backbone, seeds):
    """Formulation 2 comes from reference; 3 and 8 from setlevel; 4 and 9 from triadic."""
    log = Log("formulations_" + backbone)
    recon = load_reconstructed()
    cache = load_cache(CACHES[backbone])
    positives = load_positives(cache, 1500)
    bce = nn.BCEWithLogitsLoss()

    # Formulation 1: five seeded FIW holdouts, SWAP fidelity on against off.
    keyfn = key_for("FIW")
    for seed in range(5):
        train, test = next(iter(grouped_kfold(positives["FIW"], keyfn, FOLDS, SPLIT_SEED + seed)))
        fit_pos, val_pos = split_fit_val(train, keyfn, SPLIT_SEED + seed)
        Fit = tensors(covered(build_side(fit_pos, keyfn, negative_seed(seed)), cache), cache)
        Val = tensors(covered(build_side(val_pos, keyfn, negative_seed(seed + 100)), cache), cache)
        Test = tensors(covered(build_side(test, keyfn, negative_seed(seed + 500)), cache), cache)
        for arm in ("hybrid_q1", "hybrid_q0"):
            key = f"f1_{arm}|FIW|f0|s{seed}"
            if key in log.done:
                continue
            torch.manual_seed(seed)
            model, call, loss, lr, wd, bs, epochs = build_arm(arm)
            model.to(DEVICE)
            score = lambda m, d: batched_scores(m, d, call)
            fit_model(model, score, loss, Fit, Val, epochs, lr, wd, bs, 8, seed)
            log.write(key, arm="f1_" + arm, corpus="FIW", fold=0, seed=seed,
                      auc_matched=auc(Test, score(model, Test)))

    for corpus, pairs in positives.items():
        keyfn = key_for(corpus)
        for fold, (train, test) in enumerate(grouped_kfold(pairs, keyfn, FOLDS, SPLIT_SEED)):
            train_side = cap_pairs(covered(build_side(
                train, keyfn, negative_seed(SPLIT_SEED + fold)), cache),
                8000, negative_seed(SPLIT_SEED + fold))
            test_side = covered(build_side(
                test, keyfn, negative_seed(SPLIT_SEED + 500 + fold)), cache)
            ytr = np.array([p[2] for p in train_side])
            yte = np.array([p[2] for p in test_side])
            test_groups_a, test_groups_b = pair_groups(test_side, keyfn)

            # Formulation 5: entanglement entropy added to cosine.
            if f"f5_entropy|{corpus}|f{fold}|s0" not in log.done:
                projection = recon.fit_angle_projection(train_side, cache, 4)
                Xtr, _ = recon.entropy_pair_features(train_side, cache, projection)
                Xte, _ = recon.entropy_pair_features(test_side, cache, projection)
                # Relation columns are constant in signal terms under matched
                # negatives; dropped so no arm can use them.
                for arm, columns in (("f5_control", [0]), ("f5_entropy", [0, 1, 2])):
                    scaler = StandardScaler().fit(Xtr[:, columns])
                    clf = LogisticRegression(max_iter=4000).fit(
                        scaler.transform(Xtr[:, columns]), ytr)
                    values = clf.decision_function(scaler.transform(Xte[:, columns]))
                    log.write(f"{arm}|{corpus}|f{fold}|s0", arm=arm, corpus=corpus,
                              fold=fold, seed=0, auc_matched=float(roc_auc_score(yte, values)))
                    save_scores(f"formulations_{backbone}_{arm}_{corpus}_f{fold}_s0",
                                matched=values.astype(np.float32), y=yte.astype(np.int8),
                                groups_a=test_groups_a, groups_b=test_groups_b)

            # Formulation 6: POVM head against a matched linear head.
            for seed in seeds:
                if f"f6_povm|{corpus}|f{fold}|s{seed}" in log.done:
                    continue
                projection = recon.fit_angle_projection(train_side, cache, 1)
                Str, ytr2 = recon.amplitude_pair_states(train_side, cache, projection)
                Ste, yte2 = recon.amplitude_pair_states(test_side, cache, projection)
                for arm, factory in (("f6_povm", recon.ProjectivePOVM),
                                     ("f6_control", recon.MatchedLinearReadout)):
                    scores = recon.train_povm_model(factory(), Str, ytr2, Ste, 200,
                                                    seed + 1000 * fold, DEVICE)
                    log.write(f"{arm}|{corpus}|f{fold}|s{seed}", arm=arm, corpus=corpus,
                              fold=fold, seed=seed,
                              auc_matched=float(roc_auc_score(yte2, scores)))
                    save_scores(f"formulations_{backbone}_{arm}_{corpus}_f{fold}_s{seed}",
                                matched=np.asarray(scores, dtype=np.float32),
                                y=yte2.astype(np.int8), groups_a=test_groups_a,
                                groups_b=test_groups_b)

            # Formulation 7: purity regulariser against decorrelation.
            fit_pos, val_pos = split_fit_val(train, keyfn, SPLIT_SEED + fold)
            Fit = tensors(cap_pairs(covered(build_side(
                fit_pos, keyfn, negative_seed(SPLIT_SEED + fold)), cache),
                8000, negative_seed(SPLIT_SEED + fold)), cache)
            Val = tensors(covered(build_side(
                val_pos, keyfn, negative_seed(SPLIT_SEED + 100 + fold)), cache), cache)
            Test = tensors(test_side, cache)
            for arm in ("f7_purity", "f7_control"):
                for seed in seeds:
                    key = f"{arm}|{corpus}|f{fold}|s{seed}"
                    if key in log.done:
                        continue
                    torch.manual_seed(seed + 1000 * fold)
                    model = RelationFreeHybrid(dropout=0.5, use_quantum=True).to(DEVICE)

                    def loss(m, b, name=arm):
                        logits, h1, h2, z1, z2 = recon.hybrid_logits_and_aux(
                            m, b[0], b[1], torch.zeros_like(b[3]))
                        value = bce(logits.view(-1), b[2])
                        if name == "f7_purity":
                            purity = 0.5 * (recon.state_purity(z1, m.ent_params1)
                                            + recon.state_purity(z2, m.ent_params2))
                            return value + 0.01 * (1.0 - purity).mean()
                        return value + 0.01 * recon.decorrelation_penalty(h1, h2)
                    score = lambda m, d: batched_scores(m, d, hybrid_call(True))
                    fit_model(model, score, loss, Fit, Val, 40, 1e-3, 1e-2, 512, 8, seed)
                    values = score(model, Test)
                    log.write(key, arm=arm, corpus=corpus, fold=fold, seed=seed,
                              auc_matched=auc(Test, values))
                    save_scores(f"formulations_{backbone}_{arm}_{corpus}_f{fold}_s{seed}",
                                matched=values.astype(np.float32),
                                y=Test[2].cpu().numpy().astype(np.int8),
                                groups_a=test_groups_a, groups_b=test_groups_b)
            del Fit, Val, Test
            torch.cuda.empty_cache()


# --------------------------------------------------------------- official (Q8)
def run_rfiw(seeds):
    """Official RFIW Track-I. Test labels are used only after all fitting."""
    log = Log("rfiw_facenet")
    root = os.path.join(ROOT, "public", "rfiw_track1")
    cache = load_cache("rfiw_track1_cache.pkl")
    test_all = covered(load_labelled_pairs(os.path.join(root, "test-pairs.csv"),
                                           os.path.join(root, "test-fids")), cache)
    print("  official test pairs:", len(test_all), flush=True)
    views = {"all_types": test_all,
             "parent_child": [p for p in test_all if p[3] in PARENT_CHILD]}
    Tests = {name: tensors(pairs, cache) for name, pairs in views.items()}
    for name, pairs in views.items():
        key = f"colour|none|{name}|s0"
        if key not in log.done:
            log.write(key, arm="colour", train_on="none", view=name, seed=0,
                      auc=auc(Tests[name], colour_scores(pairs)))

    for train_on, relations in (("parent_child", set(PARENT_CHILD)), ("all_types", None)):
        fit_pos = covered(load_identity_pairs(
            os.path.join(root, "train-pairs.csv"), os.path.join(root, "train-faces"),
            relations=relations, cap_per_identity_pair=30, seed=SPLIT_SEED), cache)
        val_pos = covered(load_identity_pairs(
            os.path.join(root, "val-pairs.csv"), os.path.join(root, "val-faces"),
            relations=relations, cap_per_identity_pair=30, seed=SPLIT_SEED), cache)
        Fit = tensors(build_side(fit_pos, family_of, negative_seed(SPLIT_SEED)), cache)
        Val = tensors(build_side(val_pos, family_of, negative_seed(SPLIT_SEED + 1)), cache)
        print("  train on %s: fit %d val %d" % (train_on, len(Fit[2]), len(Val[2])), flush=True)

        def report(arm, seed, score, extra):
            for view, data in Tests.items():
                scores = score(data)
                log.write(f"{arm}|{train_on}|{view}|s{seed}", arm=arm, train_on=train_on,
                          view=view, seed=seed, auc=auc(data, scores),
                          **threshold_row(Val, score(Val), data, scores), **extra)

        if f"cosine|{train_on}|all_types|s0" not in log.done:
            report("cosine", 0, cosine_scores, {})
        if f"lr|{train_on}|all_types|s0" not in log.done:
            scaler = StandardScaler().fit(lr_features(Fit))
            clf = LogisticRegression(max_iter=500, C=1e-3).fit(
                scaler.transform(lr_features(Fit)), Fit[2].cpu().numpy())
            report("lr", 0, lambda d: clf.decision_function(scaler.transform(lr_features(d))), {})
        for arm in ("fair", "hybrid_q0"):
            for seed in seeds:
                if f"{arm}|{train_on}|all_types|s{seed}" in log.done:
                    continue
                torch.manual_seed(seed)
                model, call, loss, lr, wd, bs, epochs = build_arm(arm)
                model.to(DEVICE)
                score = lambda m, d: batched_scores(m, d, call)
                best, ran = fit_model(model, score, loss, Fit, Val, epochs, lr, wd, bs, 8, seed)
                report(arm, seed, lambda d: score(model, d), {"val_auc": best, "epochs": ran})
        del Fit, Val
        torch.cuda.empty_cache()


# ------------------------------------------------------------------- cost (Q9)
def run_cost():
    """Inference and training cost on one recorded setup, replacing 73.5x and 55x."""
    log = Log("cost")
    batch, warmup, repeats = 256, 3, 30
    a = torch.randn(batch, 512, device=DEVICE)
    b = torch.randn(batch, 512, device=DEVICE)
    r = torch.eye(4, device=DEVICE)[torch.randint(0, 4, (batch,), device=DEVICE)]
    y = torch.randint(0, 2, (batch,), device=DEVICE).float()
    bce = nn.BCEWithLogitsLoss()
    def sync():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
    def timed(step):
        for _ in range(warmup):
            step()
        out = []
        for _ in range(repeats):
            sync()
            started = time.perf_counter()
            step()
            sync()
            out.append(time.perf_counter() - started)
        return statistics.median(out)
    models = {
        "fair": (FairKinshipModel().to(DEVICE), lambda m: m.forward_logits(a, b)),
        "hybrid_q0": (RelationFreeHybrid(use_quantum=False).to(DEVICE),
                      lambda m: m.forward_logits(a, b, r)),
        "hybrid_q1": (RelationFreeHybrid(use_quantum=True).to(DEVICE),
                      lambda m: m.forward_logits(a, b, r)),
        "vqc_expectation_n6d4": (AmplitudeVQCClassifier(n_qubits_each=6, depth=4, readout="expectation").to(DEVICE),
                                 lambda m: m.forward_logits(a, b, r)),
    }
    for name, (model, forward) in models.items():
        key = f"{name}|{DEVICE.type}"
        if key in log.done:
            continue
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def train_step():
            model.train()
            optimizer.zero_grad()
            bce(forward(model).view(-1), y).backward()
            optimizer.step()
        def infer_step():
            model.eval()
            with torch.inference_mode():
                forward(model)
        inference, training = timed(infer_step), timed(train_step)
        log.write(key, arm=name, batch=batch, warmup=warmup, repeats=repeats,
                  inference_pairs_per_s=batch / inference, train_pairs_per_s=batch / training,
                  parameters=sum(p.numel() for p in model.parameters()))


def main():
    global OUT, NEGATIVE_SEED_OFFSET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["reference", "readout", "setlevel", "triadic",
                                               "formulations", "rfiw", "cost", "all"])
    parser.add_argument("--backbone", choices=sorted(CACHES), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--negative-seed-offset", type=int, default=0,
        help="offset negative-sampler seeds without changing folds or model seeds")
    parser.add_argument("--output-dir", default=OUT,
                        help="results directory; use a new directory for each run/seed offset")
    args = parser.parse_args()
    if args.negative_seed_offset < 0:
        parser.error("--negative-seed-offset must be non-negative")
    NEGATIVE_SEED_OFFSET = args.negative_seed_offset
    OUT = os.path.abspath(args.output_dir)
    backbones = [args.backbone] if args.backbone else sorted(CACHES, reverse=True)
    started = time.time()
    fast_first = ["cost", "setlevel", "triadic", "reference", "rfiw", "formulations", "readout"]
    for experiment in (fast_first if args.experiment == "all" else [args.experiment]):
        print("==", experiment, flush=True)
        if experiment == "cost":
            run_cost()
        elif experiment == "rfiw":
            run_rfiw(args.seeds)
        else:
            for backbone in backbones:
                print("  backbone", backbone, flush=True)
                if experiment == "reference":
                    run_reference(backbone, args.seeds)
                elif experiment == "readout":
                    run_readout(backbone, args.seeds)
                elif experiment == "setlevel":
                    run_setlevel(backbone)
                elif experiment == "triadic":
                    run_triadic(backbone)
                else:
                    run_formulations(backbone, args.seeds)
    print("finished in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
