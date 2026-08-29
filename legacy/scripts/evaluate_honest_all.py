# -*- coding: utf-8 -*-
"""Evaluate the deployed model on all four datasets, honestly.

FIW uses the family-disjoint held-out split. KinFaceW-I/II and TSKinFace are
fully unseen (the model never trained on them), so every pair counts as
held-out. TSKinFace uses shortcut-free negatives from src/ts_pairs.py.
"""
import json, os, pickle, sys
import numpy as np, torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import roc_auc_score

from src.calibration import operating_point, calibrate_threshold
from src.data_loaders import load_kinfacew_pairs, load_tskinface_pairs, prepare_pair_tensors, load_fiw_pairs
from src.predictor import KinshipPredictor
from src.splits import split_then_build
from src.ts_pairs import build_tskinface_pairs

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
RELS = ["fd", "fs", "md", "ms"]


def covered(pairs, cache):
    return [p for p in pairs
            if os.path.normcase(os.path.abspath(p[0])) in cache
            and os.path.normcase(os.path.abspath(p[1])) in cache]


def score(pred, pairs, cache):
    e1, e2, y, rel = prepare_pair_tensors(pairs, cache)
    rl = [RELS[i] for i in rel.argmax(1).tolist()]
    s = np.array([o["probability"] for o in pred.predict_batch(e1, e2, rl)])
    return s, y.view(-1).numpy()


def main():
    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}
    pred = KinshipPredictor(os.path.join(project_root, "weights", "deploy",
                                         "kinship_model.pt"))
    print(f"  model threshold: {pred.threshold:.4f}\n")

    datasets = {}

    kin = covered([p for p in load_fiw_pairs(os.path.join(project_root, "public"))
                   if p[2] == 1], cache)
    _, fiw_test = split_then_build(kin, test_ratio=0.2, seed=3)
    datasets["FIW (family-disjoint held-out)"] = covered(fiw_test, cache)

    for name, folder in [("KinFaceW-I", "KinFaceW-I"), ("KinFaceW-II", "KinFaceW-II")]:
        try:
            ps = load_kinfacew_pairs(os.path.join(project_root, folder))
            ps = [(a, b, l, r) for a, b, l, r, *_ in
                  [tuple(x) for x in ps]]
            datasets[f"{name} (unseen)"] = covered(ps, cache)
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    ts_root = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                           "TSKinFace_cropped")
    datasets["TSKinFace (stock pairs)"] = covered(load_tskinface_pairs(ts_root), cache)
    datasets["TSKinFace (shortcut-free)"] = covered(
        build_tskinface_pairs(ts_root, same_family_negative_ratio=0.5), cache)

    results = {}
    print(f"  {'dataset':34s} {'n':>6s} {'acc':>8s} {'AUC':>8s} {'TPR':>7s} {'FPR':>7s}")
    print("  " + "-" * 74)
    for name, pairs in datasets.items():
        if len(pairs) < 20:
            print(f"  {name:34s} {len(pairs):6d}   (too few pairs)")
            continue
        s, y = score(pred, pairs, cache)
        if len(np.unique(y)) < 2:
            print(f"  {name:34s} {len(y):6d}   (single class)")
            continue
        op = operating_point(y, s, pred.threshold)
        auc = roc_auc_score(y, s)
        # Also report the best this model could do if re-calibrated per dataset.
        t_best = calibrate_threshold(y, s, objective="accuracy")
        op_best = operating_point(y, s, t_best)
        results[name] = {"n": len(y), "accuracy": op["accuracy"], "roc_auc": float(auc),
                         "tpr": op["tpr_recall"], "fpr": op["fpr"],
                         "accuracy_recalibrated": op_best["accuracy"],
                         "threshold_recalibrated": t_best}
        print(f"  {name:34s} {len(y):6d} {op['accuracy']:7.2f}% {auc:8.4f} "
              f"{op['tpr_recall']:6.1f}% {op['fpr']:6.1f}%")

    print("\n  Re-calibrated per dataset (upper bound for this model):")
    for n, r in results.items():
        print(f"    {n:34s} {r['accuracy_recalibrated']:6.2f}%  (t={r['threshold_recalibrated']:.3f})")

    out = os.path.join(project_root, "results", "honest", "all_datasets.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n  saved -> results/honest/all_datasets.json")


if __name__ == "__main__":
    main()
