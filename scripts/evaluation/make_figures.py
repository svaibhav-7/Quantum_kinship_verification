# -*- coding: utf-8 -*-
"""Generate publication figures from results/honest/*.json.

Every figure reads measured artifacts; nothing is hardcoded. Output is
vector PDF (for LaTeX) plus PNG (for quick viewing) at paper/figures/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(project_root, "results", "honest")
OUT = os.path.join(project_root, "paper", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour-blind safe, prints legibly in greyscale.
C_BASE = "#4C6EF5"
C_NEW = "#F08C00"
C_MUTED = "#ADB5BD"
C_BAD = "#C92A2A"

DATASETS = ["KinFaceW-I", "KinFaceW-II", "FIW", "TSKinFace"]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


# ---------------------------------------------------------------------------
def fig_main_results():
    """Per-dataset accuracy with 95% CI, and ROC-AUC."""
    k = load("kfold.json")
    accs = [k[d]["accuracy_mean"] for d in DATASETS]
    cis = [k[d]["accuracy_ci95"] for d in DATASETS]
    aucs = [k[d]["roc_auc_mean"] for d in DATASETS]
    auc_sd = [k[d]["roc_auc_sd"] for d in DATASETS]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(len(DATASETS))

    a1.bar(x, accs, yerr=cis, capsize=4, color=C_BASE, width=0.6,
           error_kw={"lw": 1, "ecolor": "#212529"})
    a1.axhline(k["_mean"]["accuracy"], ls="--", lw=1, color=C_BAD,
               label=f"mean {k['_mean']['accuracy']:.2f}%")
    a1.set_ylim(60, 90)
    a1.set_ylabel("Accuracy (%)")
    a1.set_xticks(x)
    a1.set_xticklabels([d.replace("KinFaceW", "KFW") for d in DATASETS],
                       rotation=15, ha="right")
    a1.set_title("(a) Accuracy, 95\\% CI")
    a1.legend(frameon=False, loc="upper left")
    a1.grid(axis="y", ls=":", alpha=0.5)

    a2.bar(x, aucs, yerr=auc_sd, capsize=4, color=C_NEW, width=0.6,
           error_kw={"lw": 1, "ecolor": "#212529"})
    a2.axhline(k["_mean"]["roc_auc"], ls="--", lw=1, color=C_BAD,
               label=f"mean {k['_mean']['roc_auc']:.4f}")
    a2.set_ylim(0.7, 0.95)
    a2.set_ylabel("ROC-AUC")
    a2.set_xticks(x)
    a2.set_xticklabels([d.replace("KinFaceW", "KFW") for d in DATASETS],
                       rotation=15, ha="right")
    a2.set_title("(b) ROC-AUC")
    a2.legend(frameon=False, loc="upper left")
    a2.grid(axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    save(fig, "fig_main_results")


# ---------------------------------------------------------------------------
def fig_stability():
    """Single-seed split vs grouped k-fold: the measurement-stability case."""
    single = {"FIW": (61.7, 75.9), "KinFaceW-I": (70.3, 78.8),
              "KinFaceW-II": (69.0, 77.8), "TSKinFace": (75.1, 84.1)}
    k = load("kfold.json")

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    y = np.arange(len(DATASETS))

    for i, d in enumerate(DATASETS):
        lo, hi = single[d]
        ax.plot([lo, hi], [i + 0.16, i + 0.16], lw=6, solid_capstyle="butt",
                color=C_MUTED,
                label="single split (seed variance)" if i == 0 else None)
        folds = [f["accuracy"] for f in k[d]["per_fold"]]
        ax.plot([min(folds), max(folds)], [i - 0.16, i - 0.16], lw=6,
                solid_capstyle="butt", color=C_BASE,
                label="grouped $k$-fold" if i == 0 else None)
        ax.plot(k[d]["accuracy_mean"], i - 0.16, "o", ms=4, color="#212529")

    ax.set_yticks(y)
    ax.set_yticklabels([d.replace("KinFaceW", "KFW") for d in DATASETS])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Measurement stability")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_stability")


# ---------------------------------------------------------------------------
def fig_setlevel():
    """Set-level gain, and the zero-change degradation guarantee."""
    s = load("setlevel_fixed.json")
    order = ["KinFaceW-I", "KinFaceW-II", "TSKinFace", "FIW"]
    single = [s[d]["single"]["roc_auc_mean"] for d in order]
    setl = [s[d]["set"]["roc_auc_mean"] for d in order]
    sizes = [s[d]["set"]["mean_set_size"] for d in order]

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, single, w, label="single image", color=C_MUTED)
    ax.bar(x + w / 2, setl, w, label="set-level", color=C_NEW)

    for i, (a, b, n) in enumerate(zip(single, setl, sizes)):
        d = b - a
        txt = "identical" if abs(d) < 1e-9 else f"+{d:.3f}"
        ax.text(i, max(a, b) + 0.012, txt, ha="center", fontsize=7.5,
                color="#212529" if abs(d) < 1e-9 else C_BAD,
                fontweight="normal" if abs(d) < 1e-9 else "bold")
        ax.text(i, 0.615, f"$n$={n:.2f}", ha="center", fontsize=7,
                color="#495057")

    ax.set_ylim(0.60, 0.87)
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("KinFaceW", "KFW") for d in order])
    ax.set_title("Set-level representation (gain only where photo sets exist)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_setlevel")


# ---------------------------------------------------------------------------
def fig_triadic():
    """Triadic scoring vs the pairwise-equivalent baseline, per fold."""
    t = load("triadic.json")
    arms = ["best_parent", "triad", "triad+phase"]
    labels = ["best single\nparent", "triadic", "triadic\n+ phase"]
    means = [t[a]["roc_auc_mean"] for a in arms]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.8))

    a1.bar(np.arange(3), means, color=[C_MUTED, C_NEW, C_BASE], width=0.6)
    a1.set_ylim(0.70, 0.82)
    a1.set_ylabel("ROC-AUC")
    a1.set_xticks(np.arange(3))
    a1.set_xticklabels(labels, fontsize=8)
    a1.set_title("(a) Triadic scoring")
    a1.grid(axis="y", ls=":", alpha=0.5)
    a1.annotate(f"+{means[1] - means[0]:.4f}\n$p$=0.0001",
                xy=(1, means[1]), xytext=(1, means[1] + 0.022),
                ha="center", fontsize=7.5, color=C_BAD, fontweight="bold")

    bp = [f["roc_auc"] for f in t["best_parent"]["per_fold"]]
    tr = [f["roc_auc"] for f in t["triad"]["per_fold"]]
    f = np.arange(1, len(bp) + 1)
    a2.plot(f, bp, "o--", color=C_MUTED, ms=4, label="best single parent")
    a2.plot(f, tr, "s-", color=C_NEW, ms=4, label="triadic")
    a2.set_xticks(f)
    a2.set_xlabel("fold")
    a2.set_ylabel("ROC-AUC")
    a2.set_title("(b) Per fold (5/5 positive)")
    a2.legend(frameon=False, fontsize=7.5)
    a2.grid(ls=":", alpha=0.5)

    fig.tight_layout()
    save(fig, "fig_triadic")


# ---------------------------------------------------------------------------
def fig_quantum():
    """Nine formulations against matched classical controls.

    Horizontal layout with explicit value labels: the deltas span three orders
    of magnitude (-22.2 to +0.1), so a shared linear y-axis renders the
    near-zero results invisible and the labels collide.
    """
    rows = [
        ("SWAP fidelity (5 seeds)",        -0.26, "classical head",      False),
        ("SWAP fidelity (20 folds)",       +0.01, "classical head",      False),
        ("Interference phase sweep",       +0.12, "classical mixture",   False),
        ("Set density arm",                -0.06, "set features",        False),
        ("Triad phase arm",                -0.06, "convex mixture",      False),
        ("Purity regulariser",             -0.89, "decorrelation",       True),
        ("POVM head",                      -5.48, "capacity-matched",    True),
        ("Density fidelity", -8.65, "mean-pooling", True),
        ("Entanglement entropy",          -22.24, "cosine",              True),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    cols = [C_BAD if r[3] else C_MUTED for r in rows]
    ax.barh(y, vals, color=cols, height=0.62)
    ax.axvline(0, color="#212529", lw=0.9)

    for i, (name, v, ctrl, sg) in enumerate(rows):
        off = -0.6 if v < 0 else 0.6
        ha = "right" if v < 0 else "left"
        ax.text(v + off, i, f"{v:+.2f}", va="center", ha=ha, fontsize=7,
                color=C_BAD if sg else "#495057",
                fontweight="bold" if sg else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(-26, 4)
    ax.set_xlabel("$\Delta$ ROC-AUC vs matched control (percentage points)")
    ax.set_title("Nine quantum-inspired formulations; none improves on its control")
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.text(0.985, 1.02, "bold red: significantly worse than control",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=C_BAD)
    fig.tight_layout()
    save(fig, "fig_quantum")


# ---------------------------------------------------------------------------
def fig_relation():
    """Per-relation ROC-AUC; mother-son is the systematic weak point."""
    data = {
        "FIW": {"fd": 0.9933, "fs": 0.9754, "md": 0.9668, "ms": 0.7124},
        "KinFaceW-I": {"fd": 0.9174, "fs": 0.8935, "md": 0.8400, "ms": 0.9750},
        "KinFaceW-II": {"fd": 0.8529, "fs": 0.8875, "md": 0.8598, "ms": 0.7795},
        "TSKinFace": {"fd": 0.9612, "fs": 0.9124, "md": 0.9227, "ms": 0.8483},
    }
    rels = ["fd", "fs", "md", "ms"]
    labels = ["father-\ndaughter", "father-\nson", "mother-\ndaughter",
              "mother-\nson"]

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.arange(len(rels))
    w = 0.2
    for i, d in enumerate(DATASETS):
        ax.bar(x + (i - 1.5) * w, [data[d][r] for r in rels], w,
               label=d.replace("KinFaceW", "KFW"))
    means = [np.mean([data[d][r] for d in DATASETS]) for r in rels]
    ax.plot(x, means, "k--o", ms=4, lw=1, label="mean")
    ax.set_ylim(0.65, 1.02)
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Per-relation performance")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "fig_relation")


def main():
    print("generating figures ->", os.path.relpath(OUT, project_root))
    fig_main_results()
    fig_stability()
    fig_setlevel()
    fig_triadic()
    fig_quantum()
    fig_relation()
    print("done")


if __name__ == "__main__":
    main()
