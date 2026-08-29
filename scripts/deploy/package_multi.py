# -*- coding: utf-8 -*-
"""Package the multi-dataset model with per-domain thresholds.

Every threshold is calibrated on a group-disjoint VALIDATION slice of the
training side. The test split is scored once, at the end.
"""
import json, os, pickle, sys
import numpy as np, torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts", "training"))

from sklearn.metrics import roc_auc_score

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import prepare_pair_tensors
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.multi_dataset import load_all_datasets

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
WEIGHTS = os.path.join(project_root, "weights", "honest", "multi_cap8000.pt")
OUT = os.path.join(project_root, "weights", "deploy", "kinship_model.pt")


def domain_of(path):
    if "FIDs" in path: return "fiw"
    if "TSKinFace" in path: return "tskinface"
    if "KinFaceW-II" in path: return "kinfacew-ii"
    return "kinfacew-i"


def main():
    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}
    cov = lambda ps: [p for p in ps
                      if os.path.normcase(os.path.abspath(p[0])) in cache
                      and os.path.normcase(os.path.abspath(p[1])) in cache]

    from train_multi import _inner_split
    train_pairs, test_pairs, per_ds = load_all_datasets(project_root, 0.2, seed=3)
    train_pairs = cov(train_pairs)
    _, val_pairs = _inner_split(train_pairs, 3)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantumAugmentedKinshipClassifier(dropout=0.5)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.to(dev).eval()

    @torch.no_grad()
    def score(ps):
        e1, e2, y, r = prepare_pair_tensors(ps, cache)
        out = []
        for i in range(0, len(y), 2048):
            a, b, rr = e1[i:i+2048].to(dev), e2[i:i+2048].to(dev), r[i:i+2048].to(dev)
            out.append((0.5 * (model(a, b, rr) + model(b, a, rr))).view(-1).cpu())
        return torch.cat(out).numpy(), y.view(-1).numpy()

    # Global threshold from all validation data.
    vs, vy = score(cov(val_pairs))
    global_t = calibrate_threshold(vy, vs, objective="accuracy", max_fpr=0.20)

    # Per-domain thresholds from the matching validation subset.
    dom_t = {}
    for dom in ("fiw", "kinfacew-i", "kinfacew-ii", "tskinface"):
        sub = cov([p for p in val_pairs if domain_of(p[0]) == dom])
        if len(sub) < 50:
            continue
        s, y = score(sub)
        if len(np.unique(y)) < 2:
            continue
        dom_t[dom] = calibrate_threshold(y, s, objective="accuracy", max_fpr=0.25)

    print(f"  global threshold: {global_t:.4f}")
    for k, v in dom_t.items():
        print(f"    {k:14s} {v:.4f}")

    # Score test once.
    print(f"\n  {'dataset':14s} {'n':>6s} {'acc':>8s} {'AUC':>8s} {'FPR':>7s}")
    print("  " + "-" * 50)
    metrics = {}
    for name, ps in per_ds.items():
        ps = cov(ps)
        if len(ps) < 20: continue
        s, y = score(ps)
        if len(np.unique(y)) < 2: continue
        key = {"FIW": "fiw", "KinFaceW-I": "kinfacew-i",
               "KinFaceW-II": "kinfacew-ii", "TSKinFace": "tskinface"}[name]
        t = dom_t.get(key, global_t)
        op = operating_point(y, s, t)
        auc = float(roc_auc_score(y, s))
        metrics[name] = {**op, "roc_auc": auc, "n": len(y)}
        print(f"  {name:14s} {len(y):6d} {op['accuracy']:7.2f}% {auc:8.4f} {op['fpr']:6.1f}%")

    kf_path = os.path.join(project_root, "results", "honest", "kfold.json")
    kfold_acc = kfold_auc = None
    kfold_per_ds = {}
    if os.path.exists(kf_path):
        kf = json.load(open(kf_path))
        kfold_acc = kf["_mean"]["accuracy"]
        kfold_auc = kf["_mean"]["roc_auc"]
        kfold_per_ds = {k: {"accuracy": v["accuracy_mean"],
                            "ci95": v["accuracy_ci95"],
                            "roc_auc": v["roc_auc_mean"], "folds": v["folds"]}
                        for k, v in kf.items() if k != "_mean"}

    mean_acc = float(np.mean([m["accuracy"] for m in metrics.values()]))
    mean_auc = float(np.mean([m["roc_auc"] for m in metrics.values()]))
    print(f"\n  mean across datasets: {mean_acc:.2f}% acc, {mean_auc:.4f} AUC")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "threshold": global_t,
        "domain_thresholds": dom_t,
        "use_quantum": True,
        # Headline metrics come from grouped k-fold, which is the defensible
        # protocol; the single-split numbers below are kept for reference only.
        "metrics": {"accuracy": kfold_acc, "roc_auc": kfold_auc,
                    "fpr": float(np.mean([m["fpr"] for m in metrics.values()])),
                    "protocol": "grouped 5-fold, every family tested once",
                    "per_dataset_kfold": kfold_per_ds,
                    "single_split_reference": {"accuracy": mean_acc,
                                               "roc_auc": mean_auc,
                                               "per_dataset": metrics}},
        "evaluation": "group-disjoint splits on all 4 datasets; "
                      "thresholds calibrated on validation only",
    }, OUT)
    json.dump(metrics, open(os.path.join(project_root, "results", "honest",
                                         "deploy_multi.json"), "w"), indent=2)
    print(f"  saved -> {os.path.relpath(OUT, project_root)}")


if __name__ == "__main__":
    main()
