# -*- coding: utf-8 -*-
"""Package a trained checkpoint for deployment.

Calibrates the decision threshold on the family-disjoint VALIDATION split
(never on test), bundles it with the weights, and records the held-out test
metrics the model actually earned.
"""
import argparse, json, os, pickle, sys
import numpy as np, torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import accuracy_score, roc_auc_score

from src.calibration import calibrate_threshold, operating_point

from src.data_loaders import load_fiw_pairs, prepare_pair_tensors
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.splits import split_then_build, summarize_split


def load_cache(paths):
    c = {}
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                c.update({os.path.normcase(os.path.abspath(k)): v
                          for k, v in pickle.load(f).items()})
    return c


def covered(pairs, cache):
    return [p for p in pairs
            if os.path.normcase(os.path.abspath(p[0])) in cache
            and os.path.normcase(os.path.abspath(p[1])) in cache]


@torch.no_grad()
def scores(model, e1, e2, rel, dev, bs=1024):
    out = []
    for i in range(0, len(e1), bs):
        a, b = e1[i:i+bs].to(dev), e2[i:i+bs].to(dev)
        r = rel[i:i+bs].to(dev)
        out.append((0.5 * (model(a, b, r) + model(b, a, r))).view(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/honest/seed3_on.pt")
    ap.add_argument("--out", default="weights/deploy/kinship_model.pt")
    ap.add_argument("--no-quantum", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-fpr", type=float, default=0.15,
                    help="ceiling on false-positive rate at the chosen point")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_cache([
        os.path.join(project_root, "weights", "caches", "fiw_full_cache.pkl"),
        os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl"),
        os.path.join(project_root, "weights", "caches", "quick_fiw_cache.pkl"),
        os.path.join(project_root, "weights", "fiw_ensemble_emb_cache.pkl"),
    ])
    kin = covered([p for p in load_fiw_pairs(os.path.join(project_root, "public"))
                   if p[2] == 1], cache)

    train_pairs, test_pairs = split_then_build(kin, test_ratio=0.2, seed=args.seed)
    inner_pos = [p for p in train_pairs if p[2] == 1]
    _, val_pairs = split_then_build(inner_pos, test_ratio=0.2, seed=args.seed + 99)
    val_pairs, test_pairs = covered(val_pairs, cache), covered(test_pairs, cache)

    model = QuantumAugmentedKinshipClassifier(use_quantum=not args.no_quantum)
    model.load_state_dict(torch.load(os.path.join(project_root, args.weights),
                                     map_location="cpu"))
    model.to(dev).eval()

    ve1, ve2, vy, vr = prepare_pair_tensors(val_pairs, cache)
    vs = scores(model, ve1, ve2, vr, dev); vy = vy.view(-1).numpy()

    thresh = calibrate_threshold(vy, vs, objective="accuracy", max_fpr=args.max_fpr)

    te1, te2, ty, tr_ = prepare_pair_tensors(test_pairs, cache)
    ts = scores(model, te1, te2, tr_, dev); ty = ty.view(-1).numpy()
    op = operating_point(ty, ts, thresh)
    metrics = {
        "accuracy": op["accuracy"],
        "accuracy_at_0.5": float(accuracy_score(ty, (ts >= 0.5).astype(int)) * 100),
        "roc_auc": float(roc_auc_score(ty, ts)),
        "tpr_recall": op["tpr_recall"],
        "fpr": op["fpr"],
        "precision": op["precision"],
        "confusion": {k: op[k] for k in ("tp", "fp", "fn", "tn")},
        "n_test_pairs": int(len(ty)),
    }

    out = os.path.join(project_root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "threshold": thresh,
        "use_quantum": not args.no_quantum,
        "metrics": metrics,
        "split": summarize_split(train_pairs, test_pairs),
        "evaluation": "family-disjoint FIW; threshold calibrated on validation",
    }, out)

    print(f"  threshold (val-calibrated): {thresh:.4f}  (max_fpr={args.max_fpr})")
    print(f"  held-out test accuracy    : {metrics['accuracy']:.2f}%")
    print(f"  held-out test ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  recall (TPR)              : {metrics['tpr_recall']:.2f}%")
    print(f"  false-positive rate       : {metrics['fpr']:.2f}%")
    print(f"  precision                 : {metrics['precision']:.2f}%")
    print(f"  saved -> {args.out}")
    with open(os.path.join(project_root, "results", "honest", "deploy_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
