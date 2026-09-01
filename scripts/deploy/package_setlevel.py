# -*- coding: utf-8 -*-
"""Package the set-level and triadic models for deployment.

Both are fitted on a family-disjoint training split and calibrated on a
family-disjoint validation slice; the held-out fold is scored once, at the end,
and those figures are stored in the artifact so a consumer sees what the model
actually earned rather than a training-set number.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import roc_auc_score

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import load_fiw_pairs
from src.identity_sets import (build_identity_sets, features_to_vector,
                               identity_of_path, pair_set_features)
from src.kfold import balance_group_sizes, grouped_kfold
from src.set_predictor import train_set_model, train_triad_model
from src.splits import family_of
from src.triads import TRIAD_FEATURE_ORDER, find_triads, triad_features
from src.ts_pairs import family_of_ts

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
OUT_DIR = os.path.join(project_root, "weights", "deploy")
TS_ROOT = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                       "TSKinFace_cropped")


def load_cache():
    return {os.path.normcase(os.path.abspath(k)): v
            for k, v in pickle.load(open(CACHE, "rb")).items()}


def gallery_from(pairs, cache):
    out = {}
    for ident, paths in build_identity_sets(pairs).items():
        vs = [np.asarray(cache[os.path.normcase(os.path.abspath(p))])
              for p in paths if os.path.normcase(os.path.abspath(p)) in cache]
        if vs:
            out[ident] = np.stack(vs)
    return out


def id_pairs(pairs, seed):
    """Identity-level (a, b, label) with cross-family negatives built here."""
    import random

    rng = random.Random(seed)
    pos = [(identity_of_path(a), identity_of_path(b), 1) for a, b, _l, _r in pairs]
    fams = sorted({family_of(p[0]) for p in pairs})
    by_fam = {}
    for a, b, _l, _r in pairs:
        by_fam.setdefault(family_of(b), []).append(identity_of_path(b))
    out = list(pos)
    for a, _b, _l, _r in pairs:
        fa = family_of(a)
        for _ in range(16):
            fb = fams[rng.randrange(len(fams))]
            if fb != fa and by_fam.get(fb):
                out.append((identity_of_path(a),
                            by_fam[fb][rng.randrange(len(by_fam[fb]))], 0))
                break
    return out


def build_set_model(cache, seed=42):
    print("[1/2] set-level model")
    kin = balance_group_sizes(
        [p for p in load_fiw_pairs(os.path.join(project_root, "public"))
         if p[2] == 1], family_of, 600, seed)
    kin = [p for p in kin
           if os.path.normcase(os.path.abspath(p[0])) in cache
           and os.path.normcase(os.path.abspath(p[1])) in cache]

    folds = list(grouped_kfold(kin, family_of, 5, seed))
    tr_pos, te_pos = folds[0]

    # Carve a family-disjoint validation slice out of training for calibration.
    inner = list(grouped_kfold(tr_pos, family_of, 5, seed + 7))
    fit_pos, val_pos = inner[0]

    g_fit, g_val, g_te = (gallery_from(x, cache)
                          for x in (fit_pos, val_pos, te_pos))
    p_fit = id_pairs(fit_pos, seed)
    p_val = id_pairs(val_pos, seed + 1)
    p_te = id_pairs(te_pos, seed + 2)

    path = os.path.join(OUT_DIR, "set_model.pkl")
    train_set_model(g_fit, p_fit, path, seed=seed)

    m = pickle.load(open(path, "rb"))

    def score(gal, prs):
        X, y = [], []
        for a, b, lab in prs:
            A, B = gal.get(a), gal.get(b)
            if A is None or B is None:
                continue
            X.append(features_to_vector(pair_set_features(A, B)))
            y.append(lab)
        X = m["scaler"].transform(np.stack(X))
        return m["clf"].predict_proba(X)[:, 1], np.array(y)

    s_val, y_val = score(g_val, p_val)
    thr = calibrate_threshold(y_val, s_val, "accuracy", max_fpr=0.25)
    s_te, y_te = score(g_te, p_te)
    op = operating_point(y_te, s_te, thr)
    auc = float(roc_auc_score(y_te, s_te))

    sizes = [g_te[k].shape[0] for k in g_te]
    m["threshold"] = float(thr)
    m["metrics"] = {"accuracy": op["accuracy"], "roc_auc": auc,
                    "fpr": op["fpr"], "n_test_pairs": int(len(y_te)),
                    "mean_set_size": float(np.mean(sizes)),
                    "evaluation": "held-out family-disjoint fold; "
                                  "threshold calibrated on validation"}
    pickle.dump(m, open(path, "wb"))
    print("  threshold %.4f  held-out acc %.2f%%  AUC %.4f  set size %.2f"
          % (thr, op["accuracy"], auc, np.mean(sizes)))
    return m["metrics"]


def build_triad_model(cache, seed=42):
    print("[2/2] triadic model")
    if not os.path.isdir(TS_ROOT):
        print("  TSKinFace unavailable, skipping")
        return None

    triads = {f: mm for f, mm in find_triads(TS_ROOT).items()
              if all(os.path.normcase(os.path.abspath(p)) in cache
                     for p in mm.values())}
    emb = lambda p: np.asarray(cache[os.path.normcase(os.path.abspath(p))])

    import random
    rng = random.Random(seed)
    fams = sorted(triads)
    rows, groups = [], []
    for fam in fams:
        t = triads[fam]
        rows.append((emb(t["F"]), emb(t["M"]), emb(t["C"]), 1))
        groups.append(fam)
        other = triads[fams[rng.randrange(len(fams))]]
        rows.append((emb(t["F"]), emb(t["M"]), emb(other["C"]), 0))
        groups.append(fam)

    groups = np.array(groups)
    uniq = sorted(set(groups))
    rng.shuffle(uniq)
    n_te = max(1, len(uniq) // 5)
    n_val = max(1, len(uniq) // 5)
    te_f, val_f = set(uniq[:n_te]), set(uniq[n_te:n_te + n_val])

    idx = lambda pred: [i for i, g in enumerate(groups) if pred(g)]
    i_fit = idx(lambda g: g not in te_f and g not in val_f)
    i_val, i_te = idx(lambda g: g in val_f), idx(lambda g: g in te_f)

    path = os.path.join(OUT_DIR, "triad_model.pkl")
    train_triad_model([rows[i] for i in i_fit], path, seed=seed)
    m = pickle.load(open(path, "rb"))

    def score(ii):
        X = np.array([[triad_features(rows[i][0], rows[i][1], rows[i][2])[k]
                       for k in TRIAD_FEATURE_ORDER] for i in ii])
        y = np.array([rows[i][3] for i in ii])
        return m["clf"].predict_proba(m["scaler"].transform(X))[:, 1], y

    s_val, y_val = score(i_val)
    thr = calibrate_threshold(y_val, s_val, "accuracy", max_fpr=0.25)
    s_te, y_te = score(i_te)
    op = operating_point(y_te, s_te, thr)
    auc = float(roc_auc_score(y_te, s_te))

    m["threshold"] = float(thr)
    m["metrics"] = {"accuracy": op["accuracy"], "roc_auc": auc,
                    "fpr": op["fpr"], "n_test_triads": int(len(y_te)),
                    "evaluation": "held-out family-disjoint fold; "
                                  "threshold calibrated on validation"}
    pickle.dump(m, open(path, "wb"))
    print("  threshold %.4f  held-out acc %.2f%%  AUC %.4f"
          % (thr, op["accuracy"], auc))
    return m["metrics"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = load_cache()
    t0 = time.time()
    out = {"set_level": build_set_model(cache),
           "triadic": build_triad_model(cache)}
    json.dump(out, open(os.path.join(project_root, "results", "honest",
                                     "deploy_setlevel.json"), "w"), indent=2)
    print("\n  saved weights/deploy/{set_model,triad_model}.pkl  (%.0fs)"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
