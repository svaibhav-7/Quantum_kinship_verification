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
from src.models_vqc import (AmplitudeControl, AmplitudeVQCClassifier,
                            WidthMatchedControl)
from src.multi_dataset import _kfw_family
from src.splits import family_of
from src.ts_pairs import build_tskinface_pairs, family_of_ts

CACHES = {
    "facenet": os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl"),
    "arcface": os.path.join(project_root, "weights", "caches", "arcface_cache.pkl"),
}
ARMS = ("amp-expectation", "amp-fidelity", "control",
        "ctl-width-linear", "ctl-width-mlp", "ctl-width-random")


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
    if arm == "control":
        model = AmplitudeControl(n_qubits_each=args.qubits, depth=args.depth,
                                 dropout=args.dropout).to(dev)
    elif arm.startswith("ctl-width-"):
        # Matched on readout width, not just parameter count: reads out the
        # same 2n values the circuit hands its classifier.
        model = WidthMatchedControl(
            n_qubits_each=args.qubits, depth=args.depth, dropout=args.dropout,
            mode=arm.rsplit("-", 1)[1]).to(dev)
    else:
        model = AmplitudeVQCClassifier(
            n_qubits_each=args.qubits, depth=args.depth, dropout=args.dropout,
            readout="fidelity" if arm.endswith("fidelity") else "expectation"
        ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()

    T = tuple(t.to(dev) for t in prepare_pair_tensors(fit, cache))
    V = tuple(t.to(dev) for t in prepare_pair_tensors(va, cache))
    E = tuple(t.to(dev) for t in prepare_pair_tensors(te, cache))

    # Shuffled-label control: permute TRAIN labels only, leaving validation and
    # test intact. Any arm that still scores above chance is reading something
    # other than kinship (leakage, or a fold artefact), so this must come back
    # at ~0.50 for the headline result to mean anything.
    if getattr(args, "shuffle_labels", False):
        g = torch.Generator(device="cpu").manual_seed(seed + 9973)
        T = (T[0], T[1],
             T[2][torch.randperm(len(T[2]), generator=g).to(dev)],
             T[3])

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
    ap.add_argument("--backbone", choices=sorted(CACHES), default="facenet",
                    help="embedding backbone; replicating the readout ablation "
                         "across backbones tests whether it is FaceNet-specific")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="permute training labels; every arm should fall to ~0.50")
    ap.add_argument("--tag", default="amplitude_vqc")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq = AmplitudeVQCClassifier(n_qubits_each=args.qubits, depth=args.depth)
    ct = AmplitudeControl(n_qubits_each=args.qubits, depth=args.depth)
    print("  register %d qubits, dim %d, projection 512 -> %d"
          % (2 * args.qubits, 2 ** (2 * args.qubits), vq.dim))
    print("  U(theta) %d params, control head %d params  (device %s)"
          % (vq.quantum_parameter_count(), ct.head_parameter_count(), dev))

    cache_path = CACHES[args.backbone]
    print("  backbone %s  (%s)" % (args.backbone, os.path.basename(cache_path)))
    cache = {os.path.normcase(os.path.abspath(k)): val
             for k, val in pickle.load(open(cache_path, "rb")).items()}

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

    print("\n  %-13s" % "dataset" + "".join("%17s" % a for a in ARMS))
    print("  " + "-" * (13 + 17 * len(ARMS)))
    summary = {}
    for name in datasets:
        if not all(res[a][name] for a in ARMS):
            continue
        summary[name] = {}
        for a in ARMS:
            vals = [r["roc_auc"] for r in res[a][name]]
            summary[name][a] = {
                "roc_auc_mean": float(np.mean(vals)),
                "roc_auc_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "accuracy_mean": float(np.mean(
                    [r["accuracy"] for r in res[a][name]])),
                "per_fold": vals}
        print("  %-13s" % name
              + "".join("%17.4f" % summary[name][a]["roc_auc_mean"] for a in ARMS))

    if summary:
        # Bind the per-dataset rows before any bookkeeping key is added to
        # `summary`: iterating summary.values() after inserting "_test"
        # treats that dict as a dataset and raises KeyError on the arm.
        rows = list(summary.values())
        means = {a: float(np.mean([s[a]["roc_auc_mean"] for s in rows]))
                 for a in ARMS}
        print("\n  MEAN  " + "   ".join("%s %.4f" % (a, means[a]) for a in ARMS))
        allc = [x for s in rows for x in s["control"]["per_fold"]]
        tests = {}
        from scipy import stats
        for a in ARMS:
            if a == "control":
                continue
            av = [x for s in rows for x in s[a]["per_fold"]]
            if len(av) > 2:
                t, p = stats.ttest_rel(av, allc)
                print("  %-16s vs control: %+.4f  t=%.2f  p=%.4f"
                      % (a, means[a] - means["control"], t, p))
                tests[a] = {"n_folds": len(av),
                            "delta": means[a] - means["control"],
                            "t": float(t), "p": float(p)}
        summary["_test"] = tests
        summary["_mean"] = means

    summary["_config"] = {"qubits_each": args.qubits, "depth": args.depth,
                          "quantum_params": vq.quantum_parameter_count(),
                          "control_params": ct.head_parameter_count()}
    out = os.path.join(project_root, "results", "honest", args.tag + ".json")
    json.dump(summary, open(out, "w"), indent=2)
    print("\n  saved results/honest/%s.json  (%.0fs)" % (args.tag, time.time() - t0))


if __name__ == "__main__":
    main()
