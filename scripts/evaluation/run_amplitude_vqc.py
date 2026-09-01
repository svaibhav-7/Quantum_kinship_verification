# -*- coding: utf-8 -*-
"""Amplitude-encoded VQC against its matched control, full protocol.

Two arms on identical folds. The control shares the VQC's 512 -> 2**n
projection, so both see the same input capacity and the only difference is what
processes it: a variational circuit, or a classical head with at least as many
parameters.

Writes results/honest/amplitude_vqc.json.
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
from src.models_vqc import AmplitudeControl, AmplitudeVQCClassifier
from src.multi_dataset import _kfw_family
from src.splits import family_of
from src.ts_pairs import build_tskinface_pairs, family_of_ts

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")
ARMS = ("amplitude-vqc", "control")


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


def add_neg(pos, keyfn, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for a, b, _l, _r in pos:
        by[keyfn(b)].append(b)
    ks = sorted(by)
    if len(ks) < 2:
        return list(pos)
    out = list(pos)
    for a, _b, _l, rel in pos:
        for _ in range(16):
            k = ks[rng.randrange(len(ks))]
            if k != keyfn(a):
                break
        else:
            continue
        out.append((a, by[k][rng.randrange(len(by[k]))], 0, rel))
    return out


def train_eval(arm, fit, va, te, cache, args, seed, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mk = AmplitudeVQCClassifier if arm == "amplitude-vqc" else AmplitudeControl
    model = mk(n_qubits_each=args.qubits, depth=args.depth,
               dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()

    T = tuple(t.to(dev) for t in prepare_pair_tensors(fit, cache))
    V = tuple(t.to(dev) for t in prepare_pair_tensors(va, cache))
    E = tuple(t.to(dev) for t in prepare_pair_tensors(te, cache))
    vy = V[2].view(-1).cpu().numpy()

    best, best_state, since = -1.0, None, 0
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
        model.eval()
        with torch.no_grad():
            s = model.predict(V[0], V[1], V[3]).view(-1).cpu().numpy()
        a = roc_auc_score(vy, s)
        if a > best:
            best, since = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        vs = model.predict(V[0], V[1], V[3]).view(-1).cpu().numpy()
        ts = model.predict(E[0], E[1], E[3]).view(-1).cpu().numpy()
    thr = calibrate_threshold(vy, vs, "accuracy", max_fpr=0.25)
    ty = E[2].view(-1).cpu().numpy()
    op = operating_point(ty, ts, thr)
    return {"accuracy": op["accuracy"], "roc_auc": float(roc_auc_score(ty, ts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cap-per-family", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--tag", default="amplitude_vqc")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = AmplitudeVQCClassifier(n_qubits_each=args.qubits, depth=args.depth)
    ct = AmplitudeControl(n_qubits_each=args.qubits, depth=args.depth)
    print("  register %d qubits, dim %d, projection 512 -> %d"
          % (2 * args.qubits, 2 ** (2 * args.qubits), vq.dim))
    print("  U(theta) %d params, control head %d params  (device %s)"
          % (vq.quantum_parameter_count(), ct.head_parameter_count(), dev))

    cache = {os.path.normcase(os.path.abspath(k)): val
             for k, val in pickle.load(open(CACHE, "rb")).items()}

    def cov(ps):
        return [p for p in ps
                if os.path.normcase(os.path.abspath(p[0])) in cache
                and os.path.normcase(os.path.abspath(p[1])) in cache]

    datasets = {k: cov(x) for k, x in
                load_positives(args.seed, args.cap_per_family).items()}
    if args.datasets:
        datasets = {k: x for k, x in datasets.items() if k in args.datasets}

    res = {a: defaultdict(list) for a in ARMS}
    t0 = time.time()

    for name, pos in datasets.items():
        keyfn = keyfn_for(name)
        for fi, (trp, tep) in enumerate(
                grouped_kfold(pos, keyfn, args.folds, args.seed)):
            inner = list(grouped_kfold(trp, keyfn, 5, args.seed + 7))
            if not inner:
                continue
            fitp, vap = inner[0]
            fit = cov(add_neg(fitp, keyfn, args.seed + 1))
            va = cov(add_neg(vap, keyfn, args.seed + 2))
            te = cov(add_neg(tep, keyfn, args.seed + 500 + fi))
            if len(fit) < 200 or len(va) < 60 or len(te) < 60:
                continue
            line = "  %-12s fold %d:" % (name, fi + 1)
            for arm in ARMS:
                r = train_eval(arm, fit, va, te, cache, args, args.seed + fi, dev)
                res[arm][name].append(r)
                line += "  %s=%.4f" % (arm, r["roc_auc"])
            print(line, flush=True)

    print("\n  %-13s %15s %10s %9s" % ("dataset", "amplitude-vqc", "control", "delta"))
    print("  " + "-" * 50)
    summary = {}
    for name in datasets:
        v_ = [r["roc_auc"] for r in res["amplitude-vqc"][name]]
        c_ = [r["roc_auc"] for r in res["control"][name]]
        if not v_ or not c_:
            continue
        summary[name] = {
            "amplitude_vqc": {
                "roc_auc_mean": float(np.mean(v_)),
                "roc_auc_sd": float(np.std(v_, ddof=1)) if len(v_) > 1 else 0.0,
                "accuracy_mean": float(np.mean(
                    [r["accuracy"] for r in res["amplitude-vqc"][name]])),
                "per_fold": v_},
            "control": {
                "roc_auc_mean": float(np.mean(c_)),
                "roc_auc_sd": float(np.std(c_, ddof=1)) if len(c_) > 1 else 0.0,
                "accuracy_mean": float(np.mean(
                    [r["accuracy"] for r in res["control"][name]])),
                "per_fold": c_},
            "delta": float(np.mean(v_) - np.mean(c_))}
        print("  %-13s %15.4f %10.4f %+9.4f"
              % (name, np.mean(v_), np.mean(c_), np.mean(v_) - np.mean(c_)))

    if summary:
        vm = float(np.mean([s["amplitude_vqc"]["roc_auc_mean"]
                            for s in summary.values()]))
        cm = float(np.mean([s["control"]["roc_auc_mean"]
                            for s in summary.values()]))
        print("\n  MEAN  vqc %.4f   control %.4f   delta %+.4f" % (vm, cm, vm - cm))
        allv = [x for s in summary.values() for x in s["amplitude_vqc"]["per_fold"]]
        allc = [x for s in summary.values() for x in s["control"]["per_fold"]]
        if len(allv) > 2:
            from scipy import stats
            t, p = stats.ttest_rel(allv, allc)
            print("  paired t-test over %d folds: t=%.2f  p=%.4f" % (len(allv), t, p))
            summary["_test"] = {"n_folds": len(allv), "t": float(t), "p": float(p)}
        summary["_mean"] = {"amplitude_vqc": vm, "control": cm, "delta": vm - cm}

    summary["_config"] = {"qubits_each": args.qubits, "depth": args.depth,
                          "quantum_params": vq.quantum_parameter_count(),
                          "control_params": ct.head_parameter_count()}
    out = os.path.join(project_root, "results", "honest", args.tag + ".json")
    json.dump(summary, open(out, "w"), indent=2)
    print("\n  saved results/honest/%s.json  (%.0fs)" % (args.tag, time.time() - t0))


if __name__ == "__main__":
    main()
