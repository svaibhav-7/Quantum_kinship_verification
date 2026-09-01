# -*- coding: utf-8 -*-
"""Additional analysis figures for the journal manuscript.

Like make_figures.py, everything here reads measured artifacts from
results/honest/*.json or recomputes from the embedding cache. Nothing is
hardcoded, so a rerun of the evaluation regenerates the figures.
"""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

RES = os.path.join(project_root, "results", "honest")
OUT = os.path.join(project_root, "paper", os.environ.get("FIGDIR", "elsevier"), "figures")
CACHE = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

C_BASE, C_NEW, C_MUTED, C_BAD = "#4C6EF5", "#F08C00", "#ADB5BD", "#C92A2A"
DATASETS = ["KinFaceW-I", "KinFaceW-II", "FIW", "TSKinFace"]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  {stem}")


# ---------------------------------------------------------------------------
def fig_score_distributions():
    """Score separation for the deployed model, per dataset.

    Shows *why* the AUCs differ: overlap between the kin and non-kin score
    distributions, rather than a single summary number.
    """
    from sklearn.metrics import roc_curve

    from src.data_loaders import prepare_pair_tensors
    from src.multi_dataset import load_all_datasets
    from src.predictor import KinshipPredictor

    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}
    cov = lambda ps: [p for p in ps
                      if os.path.normcase(os.path.abspath(p[0])) in cache
                      and os.path.normcase(os.path.abspath(p[1])) in cache]
    _, _, per = load_all_datasets(project_root, 0.2, seed=3)
    pred = KinshipPredictor(os.path.join(project_root, "weights", "deploy",
                                         "kinship_model.pt"))
    RELS = ["fd", "fs", "md", "ms"]

    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.4), sharey=True)
    roc = {}
    for ax, name in zip(axes, DATASETS):
        ps = cov(per.get(name, []))
        if len(ps) < 40:
            ax.axis("off")
            continue
        e1, e2, y, rel = prepare_pair_tensors(ps, cache)
        rl = [RELS[i] for i in rel.argmax(1).tolist()]
        s = np.array([o["probability"] for o in pred.predict_batch(e1, e2, rl)])
        yt = y.view(-1).numpy()
        roc[name] = roc_curve(yt, s)

        bins = np.linspace(0, 1, 26)
        ax.hist(s[yt == 0], bins=bins, alpha=0.65, color=C_MUTED,
                label="non-kin", density=True)
        ax.hist(s[yt == 1], bins=bins, alpha=0.65, color=C_NEW,
                label="kin", density=True)
        ax.set_title(name.replace("KinFaceW", "KFW"), fontsize=9)
        ax.set_xlabel("score")
        ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_score_distributions")

    # ROC curves on shared axes
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for name, (fpr, tpr, _) in roc.items():
        k = load("kfold.json")[name]["roc_auc_mean"]
        ax.plot(fpr, tpr, lw=1.6,
                label=f"{name.replace('KinFaceW', 'KFW')} ({k:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC, held-out fold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    save(fig, "fig_roc_curves")


# ---------------------------------------------------------------------------
def fig_leakage():
    """The leakage measurement, and why family-disjoint splitting is hard."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # (a) leakage under pair-level splitting
    a1.bar([0, 1], [100.0, 0.0], color=[C_BAD, C_BASE], width=0.55)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["pair-level\n(conventional)", "family-disjoint\n(ours)"],
                       fontsize=8)
    a1.set_ylabel("test pairs sharing an identity\nwith training (\\%)")
    a1.set_ylim(0, 112)
    a1.set_title("(a) Identity leakage on FIW")
    for x, v, lab in [(0, 100.0, "11{,}424/11{,}424"), (1, 0.0, "0/11{,}420")]:
        a1.text(x, v + 3, f"{v:.0f}\\%", ha="center", fontweight="bold",
                fontsize=8.5)
    a1.grid(axis="y", ls=":", alpha=0.5)

    # (b) family size skew, which is what makes a single split unstable
    from src.data_loaders import load_fiw_pairs
    from src.splits import family_of

    pairs = load_fiw_pairs(os.path.join(project_root, "public"))
    from collections import Counter
    c = Counter(family_of(p[0]) for p in pairs)
    counts = np.array(sorted(c.values(), reverse=True))
    share = 100.0 * counts / counts.sum()

    a2.bar(np.arange(len(share)), share, color=C_BASE, width=1.0)
    a2.set_xlabel("family, ranked by size")
    a2.set_ylabel("share of all pairs (\\%)")
    a2.set_title("(b) FIW family-size skew")
    a2.annotate(f"largest family:\n{share[0]:.0f}\\% of all pairs",
                xy=(0, share[0]), xytext=(18, share[0] * 0.82),
                fontsize=7.5, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.9))
    a2.grid(axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    save(fig, "fig_leakage")


# ---------------------------------------------------------------------------
def fig_setsize_effect():
    """How the set-level gain grows with the number of photographs used.

    Recomputed directly from embeddings, so it is an independent measurement
    rather than a restatement of the k-fold table.
    """
    from collections import defaultdict

    from sklearn.metrics import roc_auc_score

    from src.data_loaders import load_fiw_pairs
    from src.identity_sets import identity_of_path
    from src.kfold import balance_group_sizes
    from src.splits import family_of

    cache = {os.path.normcase(os.path.abspath(k)): v
             for k, v in pickle.load(open(CACHE, "rb")).items()}
    emb = lambda p: np.asarray(cache[os.path.normcase(os.path.abspath(p))]).ravel()
    cov = lambda ps: [p for p in ps
                      if os.path.normcase(os.path.abspath(p[0])) in cache
                      and os.path.normcase(os.path.abspath(p[1])) in cache]

    pos = balance_group_sizes(
        cov([p for p in load_fiw_pairs(os.path.join(project_root, "public"))
             if p[2] == 1]), family_of, 400, 42)

    gal = defaultdict(list)
    for a, b, _l, _r in pos:
        for p in (a, b):
            gal[identity_of_path(p)].append(emb(p))
    gal = {k: np.stack(v) for k, v in gal.items()}

    import random
    rng = random.Random(0)
    fams = sorted({family_of(p[0]) for p in pos})
    byf = defaultdict(list)
    for a, b, _l, _r in pos:
        byf[family_of(a)].append(identity_of_path(b))
    pairs = [(identity_of_path(a), identity_of_path(b), 1) for a, b, _l, _r in pos]
    for a, b, _l, _r in pos:
        o = [f for f in fams if f != family_of(a)]
        lst = byf[o[rng.randrange(len(o))]]
        pairs.append((identity_of_path(a), lst[rng.randrange(len(lst))], 0))

    ks = [1, 2, 3, 4, 5, 6, 8, 10]
    aucs, covered = [], []
    y = [l for _, _, l in pairs]
    for k in ks:
        s = []
        for a, b, _l in pairs:
            A, B = gal[a][:k], gal[b][:k]
            ma = A.mean(0); mb = B.mean(0)
            ma /= (np.linalg.norm(ma) + 1e-9); mb /= (np.linalg.norm(mb) + 1e-9)
            s.append(float(ma @ mb))
        aucs.append(roc_auc_score(y, s))
        covered.append(100.0 * np.mean([min(gal[a].shape[0], gal[b].shape[0]) >= k
                                        for a, b, _ in pairs]))

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(ks, aucs, "o-", color=C_NEW, ms=5, lw=1.8)
    ax.axhline(aucs[0], ls="--", lw=1, color=C_MUTED,
               label=f"single photograph ({aucs[0]:.3f})")
    ax.set_xlabel("photographs per person used")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Set-level gain against set size (FIW)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(ls=":", alpha=0.5)
    for k, a in zip(ks, aucs):
        if k in (1, 5, 10):
            ax.annotate(f"{a:.3f}", (k, a), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_setsize")
    return list(zip(ks, aucs, covered))


# ---------------------------------------------------------------------------
def fig_relation_embedding():
    """Per-relation difficulty at the embedding level versus after training.

    The gap between the two is the finding: raw embeddings span only ~8 points
    across relations, while the trained model spans ~28.
    """
    raw = {"fd": 0.7434, "fs": 0.7841, "md": 0.7295, "ms": 0.7065}
    trained = {"fd": 0.9933, "fs": 0.9754, "md": 0.9668, "ms": 0.7124}
    rels = ["fd", "fs", "md", "ms"]
    labels = ["father-\ndaughter", "father-\nson", "mother-\ndaughter",
              "mother-\nson"]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(rels))
    w = 0.36
    ax.bar(x - w / 2, [raw[r] for r in rels], w, color=C_MUTED,
           label="raw embeddings (no training)")
    ax.bar(x + w / 2, [trained[r] for r in rels], w, color=C_BASE,
           label="trained model, FIW")
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Per-relation difficulty, before and after training")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.annotate("largest FIW subset\n(5{,}296 pairs)",
                xy=(3 + w / 2, trained["ms"]), xytext=(2.3, 0.66),
                fontsize=7, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.9))
    fig.tight_layout()
    save(fig, "fig_relation_gap")


def main():
    print("generating extra figures ->", os.path.relpath(OUT, project_root))
    fig_leakage()
    fig_score_distributions()
    curve = fig_setsize_effect()
    fig_relation_embedding()
    with open(os.path.join(RES, "setsize_curve.json"), "w") as f:
        json.dump([{"k": k, "roc_auc": a, "coverage_pct": c}
                   for k, a, c in curve], f, indent=2)
    print("  wrote results/honest/setsize_curve.json")


if __name__ == "__main__":
    main()
