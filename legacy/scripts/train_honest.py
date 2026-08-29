# -*- coding: utf-8 -*-
"""Train the quantum-augmented classifier on family-disjoint FIW splits.

Every number this produces is measured on families never seen in training.
They are not comparable to the repo's older figures, which were measured with
100% identity leakage.
"""
import argparse, json, os, pickle, sys, time
import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_loaders import load_fiw_pairs, prepare_pair_tensors, cache_face_embeddings
from src.models_hybrid import QuantumAugmentedKinshipClassifier
from src.splits import split_then_build, summarize_split


def load_cache(paths):
    cache = {}
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = pickle.load(f)
            cache.update({os.path.normcase(os.path.abspath(k)): v for k, v in raw.items()})
    return cache


def covered(pairs, cache):
    out = []
    for p in pairs:
        a = os.path.normcase(os.path.abspath(p[0]))
        b = os.path.normcase(os.path.abspath(p[1]))
        if a in cache and b in cache:
            out.append(p)
    return out


def evaluate(model, e1, e2, rel, y, bs=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(y), bs):
            outs.append(model(e1[i:i+bs], e2[i:i+bs], rel[i:i+bs]).view(-1))
    p = torch.cat(outs).cpu().numpy()
    yt = y.view(-1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(yt, (p >= 0.5).astype(int)) * 100),
        "roc_auc": float(roc_auc_score(yt, p)),
    }, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--no-quantum", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="quantum")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    print("=" * 66)
    print("  HONEST TRAINING -- family-disjoint splits")
    print(f"  quantum branch: {'OFF (ablation)' if args.no_quantum else 'ON'}")
    print(f"  device: {dev}")
    print("=" * 66)

    pairs = load_fiw_pairs(os.path.join(project_root, "public"))
    cache = load_cache([
        os.path.join(project_root, "weights", "caches", "fiw_full_cache.pkl"),
        os.path.join(project_root, "weights", "caches", "fiw_emb_cache.pkl"),
        os.path.join(project_root, "weights", "caches", "quick_fiw_cache.pkl"),
        os.path.join(project_root, "weights", "fiw_ensemble_emb_cache.pkl"),
    ])
    print(f"  cached embeddings: {len(cache)}")

    kin = [p for p in pairs if p[2] == 1]
    kin = covered(kin, cache)
    print(f"  kin pairs with embeddings: {len(kin)}")
    if args.max_pairs:
        kin = kin[:args.max_pairs]

    train_pairs, test_pairs = split_then_build(kin, test_ratio=0.2, seed=args.seed)
    train_pairs = covered(train_pairs, cache)
    test_pairs = covered(test_pairs, cache)
    s = summarize_split(train_pairs, test_pairs)
    print("\n  split report:")
    for k, v in s.items():
        print(f"    {k:20s} {v}")
    assert s["shared_identities"] == 0 and s["shared_families"] == 0, "LEAK"

    inner_pos = [p for p in train_pairs if p[2] == 1]
    train_pairs, val_pairs = split_then_build(
        inner_pos, test_ratio=args.val_ratio, seed=args.seed + 99)
    train_pairs = covered(train_pairs, cache); val_pairs = covered(val_pairs, cache)
    sv = summarize_split(train_pairs, val_pairs)
    assert sv["shared_identities"] == 0, "LEAK in inner val split"
    va = prepare_pair_tensors(val_pairs, cache)

    tr = prepare_pair_tensors(train_pairs, cache)
    te = prepare_pair_tensors(test_pairs, cache)
    tr_e1, tr_e2, tr_y, tr_rel = tr
    te_e1, te_e2, te_y, te_rel = te
    va_e1, va_e2, va_y, va_rel = va
    print(f"\n  train {len(tr_y)}  test {len(te_y)}")

    tr_e1, tr_e2, tr_y, tr_rel = (t.to(dev) for t in (tr_e1, tr_e2, tr_y, tr_rel))
    te_e1, te_e2, te_y, te_rel = (t.to(dev) for t in (te_e1, te_e2, te_y, te_rel))
    va_e1, va_e2, va_y, va_rel = (t.to(dev) for t in (va_e1, va_e2, va_y, va_rel))

    model = QuantumAugmentedKinshipClassifier(use_quantum=not args.no_quantum, dropout=args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()

    best = {"accuracy": 0.0, "roc_auc": 0.0}
    best_ep, since = 0, 0
    best_state = None
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_y), device=dev)
        tot = 0.0
        for i in range(0, len(tr_y), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            logits = model.forward_logits(tr_e1[idx], tr_e2[idx], tr_rel[idx])
            loss = lossf(logits, tr_y[idx].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        vm, _ = evaluate(model, va_e1, va_e2, va_rel, va_y)
        flag = ""
        if vm["roc_auc"] > best["roc_auc"]:
            best = vm
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep, since = ep, 0
            flag = "  <- best(val)"
        else:
            since += 1
        if ep % 2 == 0 or ep == 1 or flag:
            print(f"  epoch {ep:3d}  loss {tot/len(tr_y):.4f}   "
                  f"val acc {vm['accuracy']:.2f}%  auc {vm['roc_auc']:.4f}{flag}")
        if since >= args.patience:
            print(f"  early stop at epoch {ep} (no val gain for {args.patience})")
            break

    # Restore the epoch chosen on validation, then touch test exactly once.
    model.load_state_dict(best_state)
    test_m, _ = evaluate(model, te_e1, te_e2, te_rel, te_y)
    print(f"\n  selected epoch {best_ep} on val "
          f"(acc {best['accuracy']:.2f}%  auc {best['roc_auc']:.4f})")
    print(f"  >>> HELD-OUT TEST: acc {test_m['accuracy']:.2f}%  "
          f"auc {test_m['roc_auc']:.4f}   ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.join(project_root, "weights", "honest"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "results", "honest"), exist_ok=True)
    best_state = {k: v.cpu() for k, v in best_state.items()}
    torch.save(best_state, os.path.join(project_root, "weights", "honest", f"{args.tag}.pt"))
    with open(os.path.join(project_root, "results", "honest", f"{args.tag}.json"), "w") as f:
        json.dump({"config": vars(args), "split": s, "val_best": best,
                   "selected_epoch": best_ep, "test": test_m}, f, indent=2)
    print(f"  saved weights/honest/{args.tag}.pt")


if __name__ == "__main__":
    main()
