# -*- coding: utf-8 -*-
"""Set-level kinship evaluation under grouped k-fold.

Four arms on identical folds, so any difference is attributable to the
representation rather than to a protocol change:

  1  single      -- one photo vs one photo (current baseline)
  2  set         -- identity photo sets
  3  set+density -- set features plus Tr(rho_a rho_b), the pre-registered
                    quantum control

Reports accuracy, ROC-AUC and mean set size per dataset, so a reader can see
where set-level gains originate and where the method degrades to single-image.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import load_fiw_pairs, load_kinfacew_pairs
from src.identity_sets import (build_identity_sets, features_to_vector,
                               identity_of_path, pair_set_features)
from src.kfold import balance_group_sizes, grouped_kfold
from src.multi_dataset import _kfw_family
from src.splits import family_of
from src.ts_pairs import build_tskinface_pairs, family_of_ts

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")


def keyfn_for(name):
    if name == "FIW":
        return family_of
    if name == "TSKinFace":
        return family_of_ts
    return _kfw_family


def load_positives(seed, cap_per_family):
    out = {}
    kin = [p for p in load_fiw_pairs(os.path.join(project_root, "public"))
           if p[2] == 1]
    out["FIW"] = balance_group_sizes(kin, family_of, cap_per_family, seed)
    for folder in ("KinFaceW-I", "KinFaceW-II"):
        root = os.path.join(project_root, folder)
        if os.path.exists(root):
            ps = [(r[0], r[1], r[2], r[3]) for r in load_kinfacew_pairs(root)]
            out[folder] = [p for p in ps if p[2] == 1]
    ts_root = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                           "TSKinFace_cropped")
    if os.path.exists(ts_root):
        out["TSKinFace"] = [p for p in build_tskinface_pairs(ts_root, 0.5, seed)
                            if p[2] == 1]
    return out


def add_negatives(pos, keyfn, seed):
    """One cross-group negative per positive, drawn inside this side only.

    Rejection sampling rather than rebuilding the candidate list per pair: the
    naive version is O(n * groups) and cost 7.8 s per call on FIW, which
    dominated the whole evaluation.
    """
    import random

    rng = random.Random(seed)
    by = {}
    for a, b, _l, _r in pos:
        by.setdefault(keyfn(a), []).append(a)
        by.setdefault(keyfn(b), []).append(b)
    keys = sorted(by)
    if len(keys) < 2:
        return list(pos)

    neg = []
    n_keys = len(keys)
    for a, _b, _l, rel in pos:
        ka = keyfn(a)
        for _ in range(16):
            kb = keys[rng.randrange(n_keys)]
            if kb != ka:
                break
        else:
            continue
        pool = by[kb]
        neg.append((a, pool[rng.randrange(len(pool))], 0, rel))
    return list(pos) + neg


def build_matrix(pairs, gallery, cache, mode):
    """Feature matrix for one arm. `mode` selects the representation."""
    X, y, sizes = [], [], []
    with_density = (mode == "set+density")
    for a, b, label, _rel in pairs:
        if mode == "single":
            A = np.asarray(cache[os.path.normcase(os.path.abspath(a))])[None, :]
            B = np.asarray(cache[os.path.normcase(os.path.abspath(b))])[None, :]
        else:
            A = gallery.get(identity_of_path(a))
            B = gallery.get(identity_of_path(b))
            if A is None or B is None:
                continue
        f = pair_set_features(A, B, with_density=with_density)
        X.append(features_to_vector(f, with_density=with_density))
        y.append(label)
        sizes.append((A.shape[0] + B.shape[0]) / 2.0)
    if not X:
        return None, None, 0.0
    return np.stack(X), np.array(y), float(np.mean(sizes))


def gallery_for(pairs, cache):
    """Identity -> stacked embeddings, restricted to what the cache covers."""
    sets = build_identity_sets(pairs)
    out = {}
    for ident, paths in sets.items():
        vs = [np.asarray(cache[os.path.normcase(os.path.abspath(p))])
              for p in paths
              if os.path.normcase(os.path.abspath(p)) in cache]
        if vs:
            out[ident] = np.stack(vs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cap-per-family", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="setlevel")
    args = ap.parse_args()

    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}

    def cov(ps):
        return [p for p in ps
                if os.path.normcase(os.path.abspath(p[0])) in cache
                and os.path.normcase(os.path.abspath(p[1])) in cache]

    datasets = {k: cov(v) for k, v in
                load_positives(args.seed, args.cap_per_family).items()}

    ARMS = ("single", "set", "set+density")
    results = {arm: {d: [] for d in datasets} for arm in ARMS}
    setsize = {d: [] for d in datasets}
    t0 = time.time()

    for name, pos in datasets.items():
        keyfn = keyfn_for(name)
        print("  %s ..." % name, flush=True)
        for fi, (tr_pos, te_pos) in enumerate(
                grouped_kfold(pos, keyfn, args.folds, args.seed)):
            tr = cov(add_negatives(tr_pos, keyfn, args.seed + fi))
            te = cov(add_negatives(te_pos, keyfn, args.seed + 500 + fi))

            # Galleries are built per side: a test identity's photos never
            # inform a training feature.
            g_tr, g_te = gallery_for(tr, cache), gallery_for(te, cache)
            print("    fold %d: train %d test %d" % (fi + 1, len(tr), len(te)),
                  flush=True)

            for arm in ARMS:
                Xtr, ytr, _ = build_matrix(tr, g_tr, cache, arm)
                Xte, yte, msz = build_matrix(te, g_te, cache, arm)
                if Xtr is None or Xte is None or len(np.unique(yte)) < 2:
                    continue

                sc = StandardScaler().fit(Xtr)  # per-arm, folds are disjoint
                clf = LogisticRegression(max_iter=4000, C=1.0)
                clf.fit(sc.transform(Xtr), ytr)

                s_tr = clf.predict_proba(sc.transform(Xtr))[:, 1]
                thr = calibrate_threshold(ytr, s_tr, "accuracy", max_fpr=0.25)
                s_te = clf.predict_proba(sc.transform(Xte))[:, 1]

                op = operating_point(yte, s_te, thr)
                results[arm][name].append(
                    {"accuracy": op["accuracy"],
                     "roc_auc": float(roc_auc_score(yte, s_te)),
                     "n": int(len(yte))})
                if arm == "set":
                    setsize[name].append(msz)

    print("\n  %-14s %6s %18s %18s" % ("dataset", "arm", "accuracy", "ROC-AUC"))
    print("  " + "-" * 64)
    summary = {}
    for name in datasets:
        for arm in ARMS:
            rs = results[arm][name]
            if not rs:
                continue
            a = [r["accuracy"] for r in rs]
            u = [r["roc_auc"] for r in rs]
            ci = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
            summary.setdefault(name, {})[arm] = {
                "accuracy_mean": float(np.mean(a)),
                "accuracy_ci95": float(ci),
                "roc_auc_mean": float(np.mean(u)),
                "folds": len(a),
                "mean_set_size": float(np.mean(setsize[name])) if setsize[name] else 1.0,
            }
            print("  %-14s %6s %8.2f%% +/- %5.2f   %.4f"
                  % (name if arm == "single" else "", arm, np.mean(a), ci,
                     np.mean(u)))
        print()

    print("  %-14s %10s %10s %10s   %s" % ("dataset", "single", "set",
                                           "set+dens", "mean set size"))
    print("  " + "-" * 62)
    for name, arms in summary.items():
        row = "  %-14s" % name
        for arm in ARMS:
            row += " %9.4f" % arms[arm]["roc_auc_mean"] if arm in arms else "          "
        row += "   %.2f" % arms.get("set", {}).get("mean_set_size", 1.0)
        print(row)

    for arm in ARMS:
        vals = [summary[n][arm]["roc_auc_mean"] for n in summary if arm in summary[n]]
        acc = [summary[n][arm]["accuracy_mean"] for n in summary if arm in summary[n]]
        if vals:
            print("\n  MEAN %-12s acc %.2f%%  AUC %.4f"
                  % (arm, float(np.mean(acc)), float(np.mean(vals))))

    out = os.path.join(project_root, "results", "honest", args.tag + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print("\n  saved -> results/honest/%s.json   (%.0fs)"
          % (args.tag, time.time() - t0))


if __name__ == "__main__":
    main()
