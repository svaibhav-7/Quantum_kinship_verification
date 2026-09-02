# -*- coding: utf-8 -*-
"""Variational quantum classifier against a capacity-matched control.

Three arms on identical folds, so any difference is attributable to what
processes the angles rather than to the data or the encoder:

  vqc-fidelity     joint register, U(theta), single-scalar readout
  vqc-expectation  joint register, U(theta), per-qubit <Z> readout
  control          classical head with the same trainable parameter count

Protocol is unchanged from the rest of the project: grouped k-fold,
family-disjoint, negatives regenerated per fold, thresholds calibrated on a
group-disjoint validation slice.

Writes results/honest/vqc.json.
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
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import roc_auc_score

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import load_fiw_pairs, load_kinfacew_pairs, prepare_pair_tensors
from src.kfold import balance_group_sizes, grouped_kfold
from src.models_vqc import CapacityMatchedControl, VQCKinshipClassifier
from src.multi_dataset import _kfw_family
from src.splits import family_of
from src.ts_pairs import build_tskinface_pairs, family_of_ts

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
ARMS = ("vqc-fidelity", "vqc-expectation", "control")


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
    by = defaultdict(list)
    for a, b, _l, _r in pos:
        by[keyfn(b)].append(b)
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


def build(arm, n, depth, dropout):
    if arm == "control":
        return CapacityMatchedControl(n_qubits_each=n, depth=depth, dropout=dropout)
    readout = "fidelity" if arm.endswith("fidelity") else "expectation"
    return VQCKinshipClassifier(n_qubits_each=n, depth=depth, readout=readout,
                                dropout=dropout)


@torch.no_grad()
def score(model, e1, e2, r, bs=512):
    model.eval()
    out = []
    for i in range(0, len(e1), bs):
        out.append(model.predict(e1[i:i+bs], e2[i:i+bs], r[i:i+bs]).view(-1).cpu())
    return torch.cat(out).numpy()


def train_one(arm, tr, va, te, cache, args, seed):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(arm, args.qubits, args.depth, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()

    T = tuple(t.to(dev) for t in prepare_pair_tensors(tr, cache))
    V = tuple(t.to(dev) for t in prepare_pair_tensors(va, cache))
    E = tuple(t.to(dev) for t in prepare_pair_tensors(te, cache))
    vy = V[2].view(-1).cpu().numpy()

    best_auc, best_state, since = -1.0, None, 0
    for _ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(T[2]), device=dev)
        for i in range(0, len(perm), args.batch_size):
            j = perm[i:i + args.batch_size]
            opt.zero_grad()
            lossf(model.forward_logits(T[0][j], T[1][j], T[3][j]),
                  T[2][j].float()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if len(np.unique(vy)) < 2:
            break
        a = roc_auc_score(vy, score(model, V[0], V[1], V[3]))
        if a > best_auc:
            best_auc, since = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    vs = score(model, V[0], V[1], V[3])
    thr = calibrate_threshold(vy, vs, "accuracy", max_fpr=0.25)
    ts = score(model, E[0], E[1], E[3])
    ty = E[2].view(-1).cpu().numpy()
    op = operating_point(ty, ts, thr)
    return {"accuracy": op["accuracy"], "roc_auc": float(roc_auc_score(ty, ts)),
            "n": int(len(ty))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=4, help="qubits per person")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--cap-per-family", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--tag", default="vqc")
    args = ap.parse_args()

    q = VQCKinshipClassifier(n_qubits_each=args.qubits, depth=args.depth)
    c = CapacityMatchedControl(n_qubits_each=args.qubits, depth=args.depth)
    print(f"  joint register: {2*args.qubits} qubits, dim {2**(2*args.qubits)}")
    print(f"  U(theta) params: {q.quantum_parameter_count()}   "
          f"control head params: {c.head_parameter_count()}")

    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}
    cov = lambda ps: [p for p in ps
                      if os.path.normcase(os.path.abspath(p[0])) in cache
                      and os.path.normcase(os.path.abspath(p[1])) in cache]

    datasets = {k: cov(v) for k, v in
                load_positives(args.seed, args.cap_per_family).items()}
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    res = {a: defaultdict(list) for a in ARMS}
    t0 = time.time()

    for name, pos in datasets.items():
        keyfn = keyfn_for(name)
        for fi, (trp, tep) in enumerate(
                grouped_kfold(pos, keyfn, args.folds, args.seed)):
            tr_all = cov(add_negatives(trp, keyfn, args.seed + fi))
            te = cov(add_negatives(tep, keyfn, args.seed + 500 + fi))
            if len(tr_all) < 200 or len(te) < 60:
                continue
            # group-disjoint validation slice carved from training
            inner = list(grouped_kfold([p for p in tr_all if p[2] == 1],
                                       keyfn, 5, args.seed + 7))
            if not inner:
                continue
            fitp, vap = inner[0]
            fit = cov(add_negatives(fitp, keyfn, args.seed + 20 + fi))
            va = cov(add_negatives(vap, keyfn, args.seed + 40 + fi))
            if len(fit) < 100 or len(va) < 60:
                continue

            for arm in ARMS:
                r = train_one(arm, fit, va, te, cache, args, args.seed + fi)
                res[arm][name].append(r)
            print(f"  {name:12s} fold {fi+1}: "
                  + "  ".join(f"{a}={res[a][name][-1]['roc_auc']:.4f}"
                              for a in ARMS), flush=True)

    print(f"\n  {'dataset':13s} " + " ".join(f"{a:>16s}" for a in ARMS))
    print("  " + "-" * 66)
    summary = {}
    for name in datasets:
        row = f"  {name:13s}"
        for arm in ARMS:
            v = [r["roc_auc"] for r in res[arm][name]]
            if v:
                summary.setdefault(name, {})[arm] = {
                    "roc_auc_mean": float(np.mean(v)),
                    "roc_auc_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "accuracy_mean": float(np.mean(
                        [r["accuracy"] for r in res[arm][name]])),
                    "folds": len(v), "per_fold": v}
                row += f" {np.mean(v):16.4f}"
            else:
                row += f" {'--':>16s}"
        print(row)

    if summary:
        print("\n  mean across datasets:")
        means = {}
        for arm in ARMS:
            v = [summary[n][arm]["roc_auc_mean"] for n in summary if arm in summary[n]]
            if v:
                means[arm] = float(np.mean(v))
                print(f"    {arm:16s} {means[arm]:.4f}")
        if "control" in means:
            for arm in ("vqc-fidelity", "vqc-expectation"):
                if arm in means:
                    print(f"    {arm} minus control: {means[arm]-means['control']:+.4f}")
        summary["_mean"] = means

    summary["_config"] = {"qubits_each": args.qubits, "depth": args.depth,
                          "quantum_params": q.quantum_parameter_count(),
                          "control_params": c.head_parameter_count()}
    out = os.path.join(project_root, "results", "honest", args.tag + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n  saved results/honest/{args.tag}.json   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
