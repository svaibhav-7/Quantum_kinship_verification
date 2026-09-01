# -*- coding: utf-8 -*-
"""Sweep qubit count and circuit depth for the joint-register VQC.

One configuration is not a test of an architecture. This sweeps the two
parameters that govern its capacity -- qubits per person and circuit depth --
against the capacity-matched control at each setting, on a single fixed
family-disjoint split so the comparison is like-for-like and fast.
"""
import argparse, json, os, pickle, random, sys, time
from collections import defaultdict

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import roc_auc_score

from src.data_loaders import load_fiw_pairs, prepare_pair_tensors
from src.kfold import balance_group_sizes, grouped_kfold
from src.models_vqc import CapacityMatchedControl, VQCKinshipClassifier
from src.splits import family_of

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")


def add_neg(pos, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for a, b, _l, _r in pos:
        by[family_of(b)].append(b)
    ks = sorted(by)
    out = list(pos)
    for a, _b, _l, rel in pos:
        for _ in range(16):
            k = ks[rng.randrange(len(ks))]
            if k != family_of(a):
                break
        else:
            continue
        out.append((a, by[k][rng.randrange(len(by[k]))], 0, rel))
    return out


def train(model, T, V, epochs, bs, lr, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    vy = V[2].view(-1).cpu().numpy()
    best, best_state, since = -1.0, None, 0
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(T[2]), device=dev)
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(model.forward_logits(T[0][j], T[1][j], T[3][j]),
                  T[2][j].float()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            s = model.predict(V[0], V[1], V[3]).view(-1).cpu().numpy()
        a = roc_auc_score(vy, s)
        if a > best:
            best, since = a, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= 8:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", nargs="+", type=int, default=[4, 5, 6])
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 4, 6])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--cap-per-family", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="vqc_sweep")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}
    cov = lambda ps: [p for p in ps
                      if os.path.normcase(os.path.abspath(p[0])) in cache
                      and os.path.normcase(os.path.abspath(p[1])) in cache]

    pos = balance_group_sizes(
        cov([p for p in load_fiw_pairs(os.path.join(project_root, "public"))
             if p[2] == 1]), family_of, args.cap_per_family, args.seed)

    folds = list(grouped_kfold(pos, family_of, 5, args.seed))
    trp, tep = folds[0]
    inner = list(grouped_kfold(trp, family_of, 5, args.seed + 7))
    fitp, vap = inner[0]

    fit = cov(add_neg(fitp, 1)); va = cov(add_neg(vap, 2)); te = cov(add_neg(tep, 3))
    print(f"  fit {len(fit)}  val {len(va)}  test {len(te)}  (device {dev})")

    T = tuple(t.to(dev) for t in prepare_pair_tensors(fit, cache))
    V = tuple(t.to(dev) for t in prepare_pair_tensors(va, cache))
    E = tuple(t.to(dev) for t in prepare_pair_tensors(te, cache))
    ty = E[2].view(-1).cpu().numpy()

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            s = model.predict(E[0], E[1], E[3]).view(-1).cpu().numpy()
        return float(roc_auc_score(ty, s))

    rows = []
    print(f"\n  {'n':>2s} {'d':>2s} {'params':>7s} {'vqc-exp':>9s} {'vqc-fid':>9s} "
          f"{'control':>9s} {'best-ctl':>9s}")
    print("  " + "-" * 58)
    for n in args.qubits:
        for d in args.depths:
            t0 = time.time()
            out = {}
            for arm in ("expectation", "fidelity"):
                torch.manual_seed(args.seed)
                m = VQCKinshipClassifier(n_qubits_each=n, depth=d,
                                         readout=arm).to(dev)
                m = train(m, T, V, args.epochs, args.batch_size, args.lr, dev)
                out[arm] = evaluate(m)
            torch.manual_seed(args.seed)
            c = CapacityMatchedControl(n_qubits_each=n, depth=d).to(dev)
            c = train(c, T, V, args.epochs, args.batch_size, args.lr, dev)
            out["control"] = evaluate(c)

            nq = VQCKinshipClassifier(n_qubits_each=n, depth=d).quantum_parameter_count()
            best = max(out["expectation"], out["fidelity"]) - out["control"]
            rows.append({"n": n, "depth": d, "quantum_params": nq,
                         "secs": round(time.time() - t0, 1), **out,
                         "best_vqc_minus_control": best})
            print(f"  {n:2d} {d:2d} {nq:7d} {out['expectation']:9.4f} "
                  f"{out['fidelity']:9.4f} {out['control']:9.4f} {best:+9.4f}",
                  flush=True)

    out_path = os.path.join(project_root, "results", "honest", args.tag + ".json")
    json.dump(rows, open(out_path, "w"), indent=2)
    best = max(rows, key=lambda r: max(r["expectation"], r["fidelity"]))
    print(f"\n  best VQC config: n={best['n']} depth={best['depth']} "
          f"-> {max(best['expectation'], best['fidelity']):.4f} "
          f"(control {best['control']:.4f})")
    print(f"  saved results/honest/{args.tag}.json")


if __name__ == "__main__":
    main()
