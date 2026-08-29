# -*- coding: utf-8 -*-
"""Grouped k-fold across all four datasets.

Every family is tested exactly once. Reports mean +/- 95% CI per dataset,
replacing single-split numbers whose seed variance reached +/-6 points.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import roc_auc_score

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import prepare_pair_tensors, load_fiw_pairs, load_kinfacew_pairs
from src.kfold import grouped_kfold, balance_group_sizes
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.multi_dataset import dataset_group_key, _kfw_family
from src.splits import family_of
from src.ts_pairs import build_tskinface_pairs, family_of_ts

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
PREFIX = {"FIW": "fiw", "TSKinFace": "ts", "KinFaceW-I": "kfw1", "KinFaceW-II": "kfw2"}


def keyfn_for(name):
    if name == "FIW":
        return family_of
    if name == "TSKinFace":
        return family_of_ts
    return _kfw_family


def load_positives(cap_per_family, seed):
    out = {}
    kin = [p for p in load_fiw_pairs(os.path.join(project_root, "public")) if p[2] == 1]
    out["FIW"] = balance_group_sizes(kin, family_of, cap_per_family, seed)

    for folder in ("KinFaceW-I", "KinFaceW-II"):
        root = os.path.join(project_root, folder)
        if os.path.exists(root):
            ps = [(r[0], r[1], r[2], r[3]) for r in load_kinfacew_pairs(root)]
            out[folder] = [p for p in ps if p[2] == 1]

    ts_root = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                           "TSKinFace_cropped")
    if os.path.exists(ts_root):
        ts = build_tskinface_pairs(ts_root, 0.5, seed)
        out["TSKinFace"] = [p for p in ts if p[2] == 1]
    return out


def add_negatives(pos, keyfn, seed):
    """Build negatives inside one side so the split stays group-disjoint."""
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
    for a, _b, _l, rel in pos:
        others = [k for k in keys if k != keyfn(a)]
        pool = by[others[rng.randrange(len(others))]]
        neg.append((a, pool[rng.randrange(len(pool))], 0, rel))
    return list(pos) + neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cap-per-family", type=int, default=1500)
    ap.add_argument("--cap-per-dataset", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-quantum", action="store_true")
    ap.add_argument("--tag", default="kfold")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}

    def cov(ps):
        return [p for p in ps
                if os.path.normcase(os.path.abspath(p[0])) in cache
                and os.path.normcase(os.path.abspath(p[1])) in cache]

    datasets = {k: cov(v) for k, v in
                load_positives(args.cap_per_family, args.seed).items()}

    per_ds_folds = {}
    for name, pos in datasets.items():
        n_fam = len({keyfn_for(name)(p[0]) for p in pos})
        per_ds_folds[name] = list(grouped_kfold(pos, keyfn_for(name),
                                                args.folds, args.seed))
        print(f"  {name:14s} {len(pos):6d} positives, {n_fam:4d} families, "
              f"{len(per_ds_folds[name])} folds")

    results = {name: [] for name in datasets}
    t0 = time.time()

    for fi in range(args.folds):
        train_all, test_by_ds = [], {}
        for name, folds in per_ds_folds.items():
            if fi >= len(folds):
                continue
            tr_pos, te_pos = folds[fi]
            tr = cov(add_negatives(tr_pos, keyfn_for(name), args.seed + fi))
            te = cov(add_negatives(te_pos, keyfn_for(name), args.seed + 500 + fi))
            if args.cap_per_dataset and len(tr) > args.cap_per_dataset:
                rs = np.random.default_rng(args.seed + fi)
                tr = [tr[i] for i in rs.choice(len(tr), args.cap_per_dataset,
                                               replace=False)]
            train_all += tr
            test_by_ds[name] = te

        rs = np.random.default_rng(args.seed + fi)
        groups = sorted({dataset_group_key(p[0]) for p in train_all})
        rs.shuffle(groups)
        val_g = set(groups[:max(1, len(groups) // 6)])
        fit = [p for p in train_all
               if dataset_group_key(p[0]) not in val_g
               and dataset_group_key(p[1]) not in val_g]
        val = [p for p in train_all
               if dataset_group_key(p[0]) in val_g
               and dataset_group_key(p[1]) in val_g]
        if len(val) < 100 or len(fit) < 100:
            fit, val = train_all[:-300], train_all[-300:]

        torch.manual_seed(args.seed + fi)
        np.random.seed(args.seed + fi)
        model = QuantumAugmentedKinshipClassifier(dropout=0.5, use_quantum=not args.no_quantum).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = torch.nn.BCEWithLogitsLoss()

        ftr = tuple(t.to(dev) for t in prepare_pair_tensors(fit, cache))
        vva = tuple(t.to(dev) for t in prepare_pair_tensors(val, cache))

        @torch.no_grad()
        def sc(e1, e2, r):
            model.eval()
            out = []
            for i in range(0, len(e1), 2048):
                a, b, rr = e1[i:i + 2048], e2[i:i + 2048], r[i:i + 2048]
                out.append((0.5 * (model(a, b, rr) + model(b, a, rr))).view(-1).cpu())
            return torch.cat(out).numpy()

        best_auc, best_state, since = 0.0, None, 0
        vy = vva[2].view(-1).cpu().numpy()
        for _ep in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(ftr[2]), device=dev)
            for i in range(0, len(perm), 512):
                j = perm[i:i + 512]
                opt.zero_grad()
                lossf(model.forward_logits(ftr[0][j], ftr[1][j], ftr[3][j]),
                      ftr[2][j].float()).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            if len(np.unique(vy)) < 2:
                break
            a = roc_auc_score(vy, sc(vva[0], vva[1], vva[3]))
            if a > best_auc:
                best_auc, since = a, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                since += 1
                if since >= 8:
                    break
        if best_state:
            model.load_state_dict(best_state)

        vs = sc(vva[0], vva[1], vva[3])
        line = "  fold %d:" % (fi + 1)
        for name, te in test_by_ds.items():
            if len(te) < 20:
                continue
            e1, e2, y, r = prepare_pair_tensors(te, cache)
            s = sc(e1.to(dev), e2.to(dev), r.to(dev))
            yt = y.view(-1).numpy()
            if len(np.unique(yt)) < 2:
                continue

            # Threshold from the matching slice of validation, never from test.
            sub = [p for p in val
                   if dataset_group_key(p[0]).startswith(PREFIX[name])]
            t = None
            if len(sub) >= 50:
                se1, se2, sy, sr = prepare_pair_tensors(sub, cache)
                syt = sy.view(-1).numpy()
                if len(np.unique(syt)) > 1:
                    ss = sc(se1.to(dev), se2.to(dev), sr.to(dev))
                    t = calibrate_threshold(syt, ss, "accuracy", max_fpr=0.25,
                                            smooth=True)
            if t is None:
                t = calibrate_threshold(vy, vs, "accuracy", max_fpr=0.25,
                                        smooth=True)

            op = operating_point(yt, s, t)
            results[name].append({"accuracy": op["accuracy"],
                                  "roc_auc": float(roc_auc_score(yt, s)),
                                  "fpr": op["fpr"], "n": int(len(yt))})
            line += "  %s=%.1f%%" % (name, op["accuracy"])
        print(line)

    print("\n  %-14s %5s %18s %18s" % ("dataset", "folds", "accuracy", "ROC-AUC"))
    print("  " + "-" * 60)
    summary = {}
    for name, rs_ in results.items():
        if not rs_:
            continue
        a = [r["accuracy"] for r in rs_]
        u = [r["roc_auc"] for r in rs_]
        sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        ci = 1.96 * sd / np.sqrt(len(a)) if len(a) > 1 else 0.0
        summary[name] = {"folds": len(a), "accuracy_mean": float(np.mean(a)),
                         "accuracy_sd": sd, "accuracy_ci95": float(ci),
                         "roc_auc_mean": float(np.mean(u)),
                         "roc_auc_sd": float(np.std(u, ddof=1)) if len(u) > 1 else 0.0,
                         "per_fold": rs_}
        print("  %-14s %5d %8.2f%% +/- %5.2f   %.4f +/- %.4f"
              % (name, len(a), np.mean(a), ci, np.mean(u),
                 np.std(u, ddof=1) if len(u) > 1 else 0.0))

    if summary:
        ma = float(np.mean([v["accuracy_mean"] for v in summary.values()]))
        mu = float(np.mean([v["roc_auc_mean"] for v in summary.values()]))
        print("\n  MEAN ACROSS DATASETS: %.2f%% accuracy, %.4f ROC-AUC" % (ma, mu))
        summary["_mean"] = {"accuracy": ma, "roc_auc": mu}

    out = os.path.join(project_root, "results", "honest", args.tag + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print("  saved -> results/honest/%s.json   (%.0fs)" % (args.tag, time.time() - t0))


if __name__ == "__main__":
    main()
