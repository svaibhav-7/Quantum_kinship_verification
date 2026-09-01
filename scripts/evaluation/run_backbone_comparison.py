# -*- coding: utf-8 -*-
"""Do the paper's findings hold under a second backbone?

Three claims are re-tested with ArcFace (buffalo_l, w600k_r50) substituted for
FaceNet, everything else identical:

  1. leakage        -- a property of the split, so it must be backbone-invariant
  2. set-level gain -- does person-level aggregation still help?
  3. quantum arm    -- does the density fidelity still fail?

Agreement across two backbones trained with different objectives on different
corpora is what separates a property of the task from an artefact of one
embedding.
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_loaders import load_fiw_pairs, load_kinfacew_pairs
from src.identity_sets import (build_identity_sets, features_to_vector,
                               identity_of_path, pair_set_features)
from src.kfold import balance_group_sizes, grouped_kfold
from src.multi_dataset import _kfw_family
from src.splits import family_of
from src.ts_pairs import build_tskinface_pairs, family_of_ts

CACHES = {
    "FaceNet": os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl"),
    "ArcFace": os.path.join(project_root, "weights", "caches", "arcface_cache.pkl"),
}


def keyfn_for(name):
    if name == "FIW":
        return family_of
    if name == "TSKinFace":
        return family_of_ts
    return _kfw_family


def load_positives(seed, cap):
    out = {}
    kin = [p for p in load_fiw_pairs(os.path.join(project_root, "public"))
           if p[2] == 1]
    out["FIW"] = balance_group_sizes(kin, family_of, cap, seed)
    for folder in ("KinFaceW-I", "KinFaceW-II"):
        root = os.path.join(project_root, folder)
        if os.path.exists(root):
            ps = [(r[0], r[1], r[2], r[3]) for r in load_kinfacew_pairs(root)]
            out[folder] = [p for p in ps if p[2] == 1]
    ts = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                      "TSKinFace_cropped")
    if os.path.exists(ts):
        out["TSKinFace"] = [p for p in build_tskinface_pairs(ts, 0.5, seed)
                            if p[2] == 1]
    return out


def add_negatives(pos, keyfn, seed):
    rng = random.Random(seed)
    by = {}
    for a, b, _l, _r in pos:
        by.setdefault(keyfn(b), []).append(b)
    keys = sorted(by)
    if len(keys) < 2:
        return list(pos)
    out = list(pos)
    for a, _b, _l, rel in pos:
        for _ in range(16):
            k = keys[rng.randrange(len(keys))]
            if k != keyfn(a):
                break
        else:
            continue
        out.append((a, by[k][rng.randrange(len(by[k]))], 0, rel))
    return out


def gallery(pairs, cache):
    out = {}
    for ident, paths in build_identity_sets(pairs).items():
        vs = [np.asarray(cache[os.path.normcase(os.path.abspath(p))])
              for p in paths if os.path.normcase(os.path.abspath(p)) in cache]
        if vs:
            out[ident] = np.stack(vs)
    return out


def matrix(pairs, gal, cache, mode):
    X, y = [], []
    dens = mode == "set+density"
    for a, b, label, _rel in pairs:
        if mode == "single":
            A = np.asarray(cache[os.path.normcase(os.path.abspath(a))])[None, :]
            B = np.asarray(cache[os.path.normcase(os.path.abspath(b))])[None, :]
        else:
            A, B = gal.get(identity_of_path(a)), gal.get(identity_of_path(b))
            if A is None or B is None:
                continue
        X.append(features_to_vector(pair_set_features(A, B, with_density=dens),
                                    with_density=dens))
        y.append(label)
    if not X:
        return None, None
    return np.stack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cap-per-family", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ARMS = ("single", "set", "set+density")
    summary = {}
    t0 = time.time()

    for bb, path in CACHES.items():
        if not os.path.exists(path):
            print(f"  [skip] {bb}: cache missing")
            continue
        cache = {os.path.normcase(os.path.abspath(k)): v
                 for k, v in pickle.load(open(path, "rb")).items()}
        cov = lambda ps: [p for p in ps
                          if os.path.normcase(os.path.abspath(p[0])) in cache
                          and os.path.normcase(os.path.abspath(p[1])) in cache]
        print(f"\n=== {bb} ({len(cache)} embeddings) ===", flush=True)

        datasets = {k: cov(v) for k, v in
                    load_positives(args.seed, args.cap_per_family).items()}
        res = {a: defaultdict(list) for a in ARMS}

        for name, pos in datasets.items():
            keyfn = keyfn_for(name)
            for fi, (trp, tep) in enumerate(
                    grouped_kfold(pos, keyfn, args.folds, args.seed)):
                tr = cov(add_negatives(trp, keyfn, args.seed + fi))
                te = cov(add_negatives(tep, keyfn, args.seed + 500 + fi))
                g_tr, g_te = gallery(tr, cache), gallery(te, cache)
                for arm in ARMS:
                    Xtr, ytr = matrix(tr, g_tr, cache, arm)
                    Xte, yte = matrix(te, g_te, cache, arm)
                    if Xtr is None or Xte is None or len(np.unique(yte)) < 2:
                        continue
                    sc = StandardScaler().fit(Xtr)
                    clf = LogisticRegression(max_iter=4000).fit(sc.transform(Xtr), ytr)
                    s = clf.predict_proba(sc.transform(Xte))[:, 1]
                    res[arm][name].append(float(roc_auc_score(yte, s)))
            print(f"  {name:12s} done", flush=True)

        summary[bb] = {arm: {d: {"mean": float(np.mean(v)),
                                 "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                 "folds": len(v)}
                             for d, v in res[arm].items()}
                       for arm in ARMS}

    print(f"\n  {'dataset':13s} {'FaceNet single':>15s} {'FaceNet set':>12s} "
          f"{'ArcFace single':>15s} {'ArcFace set':>12s}")
    print("  " + "-" * 72)
    for d in ("KinFaceW-I", "KinFaceW-II", "FIW", "TSKinFace"):
        row = f"  {d:13s}"
        for bb in ("FaceNet", "ArcFace"):
            for arm in ("single", "set"):
                v = summary.get(bb, {}).get(arm, {}).get(d)
                row += f" {v['mean']:14.4f}" if v else f" {'--':>14s}"
        print(row)

    print("\n  set-level gain by backbone:")
    for bb in summary:
        for d in ("FIW",):
            a = summary[bb]["single"].get(d)
            b = summary[bb]["set"].get(d)
            if a and b:
                print(f"    {bb:8s} {d}: {a['mean']:.4f} -> {b['mean']:.4f}  "
                      f"({b['mean']-a['mean']:+.4f})")

    print("\n  quantum density arm (set+density minus set):")
    for bb in summary:
        for d in ("KinFaceW-I", "KinFaceW-II", "FIW", "TSKinFace"):
            a = summary[bb]["set"].get(d)
            b = summary[bb]["set+density"].get(d)
            if a and b:
                print(f"    {bb:8s} {d:12s} {b['mean']-a['mean']:+.4f}")

    out = os.path.join(project_root, "results", "honest", "backbone_comparison.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n  saved results/honest/backbone_comparison.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
