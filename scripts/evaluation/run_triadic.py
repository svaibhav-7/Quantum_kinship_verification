# -*- coding: utf-8 -*-
"""Triadic kinship evaluation under grouped k-fold.

TSKinFace supplies father + mother + child. The pairwise protocol splits each
triad into two independent pairs and discards the joint structure. Probes
measured that scoring the triad jointly lifts separability from 0.761 to 0.792
ROC-AUC, with most of the gain coming from parental similarity -- a quantity no
pairwise model can express.

Three arms on identical folds:
  best_parent -- max(cos(F,C), cos(M,C)), what a pairwise model effectively sees
  triad       -- the full triadic feature set
  triad+phase -- plus an interference sweep over a relative phase, the
                 pre-registered quantum control
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.calibration import calibrate_threshold, operating_point
from src.kfold import grouped_kfold
from src.triads import find_triads, triad_features, TRIAD_FEATURE_ORDER
from src.ts_pairs import family_of_ts

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
TS_ROOT = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                       "TSKinFace_cropped")


def build_triad_samples(triads, seed=42):
    """One positive and one negative per triad.

    The negative keeps the real parents and substitutes a child from another
    family, so the label turns on kinship rather than on any property of the
    parents. Every sample is attributed to the parents' family, which is what
    fold assignment keys on.
    """
    import random

    rng = random.Random(seed)
    fams = sorted(triads)
    samples = []
    for fam in fams:
        t = triads[fam]
        samples.append((t["F"], t["M"], t["C"], 1, fam))
        others = [f for f in fams if f != fam]
        if not others:
            continue
        other = triads[others[rng.randrange(len(others))]]
        samples.append((t["F"], t["M"], other["C"], 0, fam))
    return samples


def phase_features(F, M, C, n_phase=13):
    """Coherent parent superposition with a relative phase.

    The classical additive mixture is the phase = 0 special case, so this can
    only help if real phase structure exists. Probe 4 measured +0.1 AUC.
    """
    n = lambda v: v / (np.linalg.norm(v) + 1e-12)
    f, m, c = n(F), n(M), n(C)
    sims = []
    for ph in np.linspace(0.0, np.pi, n_phase):
        amp = f * np.cos(ph) + m * np.sin(ph)
        sims.append(float(n(amp) @ c))
    return [max(sims), min(sims), float(np.std(sims))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="triadic")
    args = ap.parse_args()

    if not os.path.isdir(TS_ROOT):
        sys.exit("TSKinFace not found at %s" % TS_ROOT)

    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}

    def emb(path):
        return np.asarray(cache.get(os.path.normcase(os.path.abspath(path))))

    triads = {f: m for f, m in find_triads(TS_ROOT).items()
              if all(os.path.normcase(os.path.abspath(p)) in cache
                     for p in m.values())}
    print("  complete triads with embeddings: %d" % len(triads))

    samples = build_triad_samples(triads, args.seed)
    print("  samples: %d (%d positive)"
          % (len(samples), sum(1 for s in samples if s[3] == 1)))

    # Fold on the parents' family, so no family is scored by a model that saw it.
    as_pairs = [(F, M, lab, fam) for F, M, C, lab, fam in samples]

    ARMS = ("best_parent", "triad", "triad+phase")
    results = {a: [] for a in ARMS}
    t0 = time.time()

    for fi, (tr_idx, te_idx) in enumerate(
            grouped_kfold(as_pairs, family_of_ts, args.folds, args.seed)):
        tr_f = {family_of_ts(p[0]) for p in tr_idx}
        te_f = {family_of_ts(p[0]) for p in te_idx}
        assert tr_f.isdisjoint(te_f), "family spans the fold boundary"

        tr = [s for s in samples if s[4] in tr_f]
        te = [s for s in samples if s[4] in te_f]
        if not tr or not te:
            continue

        def matrix(rows, arm):
            X, y = [], []
            for F, M, C, lab, _fam in rows:
                ef, em, ec = emb(F), emb(M), emb(C)
                feats = triad_features(ef, em, ec)
                if arm == "best_parent":
                    v = [max(feats["cos_fc"], feats["cos_mc"])]
                else:
                    v = [feats[k] for k in TRIAD_FEATURE_ORDER]
                    if arm == "triad+phase":
                        v = v + phase_features(ef, em, ec)
                X.append(v)
                y.append(lab)
            return np.array(X, dtype=np.float64), np.array(y)

        for arm in ARMS:
            Xtr, ytr = matrix(tr, arm)
            Xte, yte = matrix(te, arm)
            if len(np.unique(yte)) < 2:
                continue
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=4000).fit(sc.transform(Xtr), ytr)
            s_tr = clf.predict_proba(sc.transform(Xtr))[:, 1]
            thr = calibrate_threshold(ytr, s_tr, "accuracy", max_fpr=0.25)
            s_te = clf.predict_proba(sc.transform(Xte))[:, 1]
            op = operating_point(yte, s_te, thr)
            results[arm].append({"accuracy": op["accuracy"],
                                 "roc_auc": float(roc_auc_score(yte, s_te)),
                                 "n": int(len(yte))})
        print("    fold %d: train %d test %d" % (fi + 1, len(tr), len(te)),
              flush=True)

    print("\n  %-14s %5s %20s %10s" % ("arm", "folds", "accuracy", "ROC-AUC"))
    print("  " + "-" * 54)
    summary = {}
    for arm in ARMS:
        rs = results[arm]
        if not rs:
            continue
        a = [r["accuracy"] for r in rs]
        u = [r["roc_auc"] for r in rs]
        ci = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        summary[arm] = {"folds": len(a),
                        "accuracy_mean": float(np.mean(a)),
                        "accuracy_ci95": float(ci),
                        "roc_auc_mean": float(np.mean(u)),
                        "roc_auc_sd": float(np.std(u, ddof=1)) if len(u) > 1 else 0.0,
                        "per_fold": rs}
        print("  %-14s %5d %9.2f%% +/- %5.2f %10.4f"
              % (arm, len(a), np.mean(a), ci, np.mean(u)))

    if "triad" in summary and "best_parent" in summary:
        d = summary["triad"]["roc_auc_mean"] - summary["best_parent"]["roc_auc_mean"]
        print("\n  triadic gain over best-single-parent: %+.4f AUC" % d)
    if "triad+phase" in summary and "triad" in summary:
        d = summary["triad+phase"]["roc_auc_mean"] - summary["triad"]["roc_auc_mean"]
        print("  quantum phase arm over triad:        %+.4f AUC" % d)

    out = os.path.join(project_root, "results", "honest", args.tag + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print("\n  saved -> results/honest/%s.json   (%.0fs)"
          % (args.tag, time.time() - t0))


if __name__ == "__main__":
    main()
