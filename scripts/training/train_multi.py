# -*- coding: utf-8 -*-
"""Train on all four datasets pooled, with per-dataset held-out reporting."""
import argparse, json, os, pickle, sys, time
import numpy as np, torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import roc_auc_score

from src.calibration import calibrate_threshold, operating_point
from src.data_loaders import prepare_pair_tensors
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.multi_dataset import load_all_datasets

CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")


from src.multi_dataset import dataset_group_key as _group_key


def _inner_split(pairs, seed, val_ratio=0.15):
    import random
    groups = sorted({_group_key(p[0]) for p in pairs} | {_group_key(p[1]) for p in pairs})
    rng = random.Random(seed + 555); rng.shuffle(groups)
    val_g = set(groups[:max(1, int(len(groups) * val_ratio))])
    fit, val = [], []
    for p in pairs:
        ga, gb = _group_key(p[0]), _group_key(p[1])
        if ga in val_g and gb in val_g:
            val.append(p)
        elif ga not in val_g and gb not in val_g:
            fit.append(p)
    return fit, val


def covered(pairs, cache):
    return [p for p in pairs
            if os.path.normcase(os.path.abspath(p[0])) in cache
            and os.path.normcase(os.path.abspath(p[1])) in cache]


@torch.no_grad()
def scores(model, e1, e2, rel, dev, bs=2048):
    model.eval(); out = []
    for i in range(0, len(e1), bs):
        a, b, r = e1[i:i+bs].to(dev), e2[i:i+bs].to(dev), rel[i:i+bs].to(dev)
        out.append((0.5 * (model(a, b, r) + model(b, a, r))).view(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-quantum", action="store_true")
    ap.add_argument("--tag", default="multi")
    ap.add_argument("--cap-per-dataset", type=int, default=0,
                    help="max training pairs per dataset (0 = no cap)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}

    train_pairs, test_pairs, per_ds = load_all_datasets(
        project_root, test_ratio=0.2, seed=args.seed)
    train_pairs = covered(train_pairs, cache)
    test_pairs = covered(test_pairs, cache)
    per_ds = {k: covered(v, cache) for k, v in per_ds.items()}

    # Inner validation must be group-disjoint too: a random slice of pooled
    # pairs shares families with the fit set and inflated val AUC to 0.995
    # while true test AUC was 0.844.
    fit_pairs, val_pairs = _inner_split(train_pairs, args.seed)

    if args.cap_per_dataset:
        from collections import defaultdict
        buckets = defaultdict(list)
        for p in fit_pairs:
            tagk = ("FIW" if "FIDs" in p[0]
                    else "TS" if "TSKinFace" in p[0] else "KFW")
            buckets[tagk].append(p)
        rng2 = np.random.default_rng(args.seed)
        capped = []
        for k, v in buckets.items():
            if len(v) > args.cap_per_dataset:
                sel = rng2.choice(len(v), args.cap_per_dataset, replace=False)
                v = [v[i] for i in sel]
            capped.extend(v)
            print(f"    fit/{k:4s} {len(v)}")
        fit_pairs = capped

    print(f"  train {len(fit_pairs)}  val {len(val_pairs)}  test {len(test_pairs)}")
    for k, v in per_ds.items():
        print(f"    test/{k:12s} {len(v)}")

    tr = prepare_pair_tensors(fit_pairs, cache)
    va = prepare_pair_tensors(val_pairs, cache)
    tr = tuple(t.to(dev) for t in tr); va_cpu = va
    va = tuple(t.to(dev) for t in va)

    model = QuantumAugmentedKinshipClassifier(
        use_quantum=not args.no_quantum, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()

    best_auc, best_state, best_ep, since = 0.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr[2]), device=dev)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            j = perm[i:i+args.batch_size]
            opt.zero_grad()
            lg = model.forward_logits(tr[0][j], tr[1][j], tr[3][j])
            loss = lossf(lg, tr[2][j].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(j)
        sched.step()

        vs = scores(model, va[0], va[1], va[3], dev)
        vy = va[2].view(-1).cpu().numpy()
        auc = roc_auc_score(vy, vs)
        flag = ""
        if auc > best_auc:
            best_auc, best_ep, since = auc, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            flag = "  <- best(val)"
        else:
            since += 1
        if ep % 5 == 0 or ep == 1 or flag:
            print(f"  epoch {ep:3d}  loss {tot/len(perm):.4f}  val auc {auc:.4f}{flag}")
        if since >= args.patience:
            print(f"  early stop at {ep}"); break

    model.load_state_dict(best_state)

    vs = scores(model, va[0], va[1], va[3], dev)
    vy = va[2].view(-1).cpu().numpy()
    thresh = calibrate_threshold(vy, vs, objective="accuracy", max_fpr=0.15, smooth=True)

    print(f"\n  selected epoch {best_ep}, threshold {thresh:.4f}")
    print(f"  {'dataset':22s} {'n':>6s} {'acc':>8s} {'AUC':>8s} {'FPR':>7s}")
    print("  " + "-" * 56)
    results = {}
    for name, ps in [("ALL (pooled)", test_pairs)] + list(per_ds.items()):
        if len(ps) < 20: continue
        e1, e2, y, rel = prepare_pair_tensors(ps, cache)
        s = scores(model, e1, e2, rel, dev); yt = y.view(-1).numpy()
        if len(np.unique(yt)) < 2: continue
        op = operating_point(yt, s, thresh); auc = float(roc_auc_score(yt, s))
        results[name] = {**op, "roc_auc": auc, "n": len(yt)}
        print(f"  {name:22s} {len(yt):6d} {op['accuracy']:7.2f}% {auc:8.4f} {op['fpr']:6.1f}%")

    os.makedirs(os.path.join(project_root, "weights", "honest"), exist_ok=True)
    torch.save({k: v.cpu() for k, v in best_state.items()},
               os.path.join(project_root, "weights", "honest", f"{args.tag}.pt"))
    json.dump({"config": vars(args), "threshold": thresh, "results": results},
              open(os.path.join(project_root, "results", "honest", f"{args.tag}.json"), "w"),
              indent=2)
    print(f"\n  saved weights/honest/{args.tag}.pt  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
