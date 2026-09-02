# -*- coding: utf-8 -*-
"""Figures for the readout-bottleneck result.

Three figures, all read from results/honest/*.json; nothing is hardcoded.

  fig_architecture   the pipeline, drawn to show where the two readouts branch
  fig_readout        per-corpus vector-vs-scalar across both backbones
  fig_readout_sweep  ROC-AUC against number of retained observables

Style follows make_figures.py so the set reads as one system.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(project_root, "results", "honest")
OUT = os.path.join(project_root, "paper", "elsevier", "figures")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

C_BASE = "#4C6EF5"   # scalar / baseline
C_NEW = "#F08C00"    # vector / the effect
C_MUTED = "#ADB5BD"
C_BAD = "#C92A2A"
NL = chr(10)
PHI = "phi(.)"
SCALAR_LBL = "scalar fidelity" + NL + "1 value"
VECTOR_LBL = "expectation vector" + NL + "2n values"
CORPORA = ["FIW", "KinFaceW-I", "KinFaceW-II", "TSKinFace"]


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


# ---------------------------------------------------------------------------
def fig_architecture():
    """The pipeline, with the readout branch drawn as the manipulated variable.

    Deliberately not a generic block diagram: the two measurement branches are
    the experiment, so they are the visual focus and everything shared between
    the arms is drawn in muted grey to show it is held fixed.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=C_MUTED, tc="black", lw=1.0, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.35)

    def arrow(x1, y1, x2, y2, c=C_MUTED, lw=1.0, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=9, color=c, lw=lw, zorder=1))

    # ---- shared trunk: identical across every arm -------------------------
    box(0.15, 3.55, 1.15, 0.85, "X1" + NL + "photo", fc="#F1F3F5")
    box(0.15, 1.30, 1.15, 0.85, "X2" + NL + "photo", fc="#F1F3F5")
    box(1.75, 3.55, 1.35, 0.85, "frozen" + NL + PHI, fc="#F1F3F5")
    box(1.75, 1.30, 1.35, 0.85, "frozen" + NL + PHI, fc="#F1F3F5")
    box(3.55, 3.55, 1.30, 0.85, "amplitude\nencode", fc="#F1F3F5")
    box(3.55, 1.30, 1.30, 0.85, "amplitude\nencode", fc="#F1F3F5")
    for y in (3.975, 1.725):
        arrow(1.30, y, 1.75, y); arrow(3.10, y, 3.55, y)

    # joint register + circuit
    box(5.30, 2.35, 1.45, 1.20,
        "joint register" + NL + "dim 2^(2n)" + NL + "U(theta)", fc="#E7F5FF", ec=C_BASE, lw=1.2)
    arrow(4.85, 3.975, 5.55, 3.55, c=C_BASE)
    arrow(4.85, 1.725, 5.55, 2.35, c=C_BASE)

    # ---- the manipulated variable ----------------------------------------
    box(7.25, 3.35, 2.55, 0.95,
        SCALAR_LBL,
        fc="#E7F5FF", ec=C_BASE, lw=1.4, fs=8)
    box(7.25, 1.45, 2.55, 0.95,
        VECTOR_LBL,
        fc="#FFF4E6", ec=C_NEW, lw=1.8, fs=8)
    arrow(6.75, 3.20, 7.25, 3.80, c=C_BASE, lw=1.3)
    arrow(6.75, 2.70, 7.25, 1.95, c=C_NEW, lw=1.8)

    ax.text(6.62, 4.72, "only this differs between arms", fontsize=8.5,
            style="italic", color=C_NEW, ha="center")
    ax.add_patch(FancyBboxPatch((7.05, 1.25), 2.95, 3.20,
                                boxstyle="round,pad=0.05,rounding_size=0.10",
                                fc="none", ec=C_NEW, lw=1.0, ls=(0, (4, 3)),
                                zorder=0))
    ax.text(2.5, 0.55, "shared and held fixed: encoder, circuit, folds, "
                       "optimiser, parameter budget",
            fontsize=8, color=C_MUTED, ha="center", style="italic")
    save(fig, "fig_architecture")


# ---------------------------------------------------------------------------
def fig_readout():
    """Per-corpus vector vs scalar, both backbones, from readout_summary.json."""
    d = json.load(open(os.path.join(RES, "readout_summary.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    x = np.arange(len(CORPORA)); w = 0.36

    for ax, bb in zip(axes, ("FaceNet", "ArcFace")):
        sc = [d["per_cell"][bb][c]["scalar"] for c in CORPORA]
        ve = [d["per_cell"][bb][c]["vector"] for c in CORPORA]
        ax.bar(x - w/2, sc, w, label=SCALAR_LBL, color=C_BASE)
        ax.bar(x + w/2, ve, w, label=VECTOR_LBL, color=C_NEW)
        for i, (s, v) in enumerate(zip(sc, ve)):
            ax.annotate(f"+{v-s:.3f}", (i + w/2, v + 0.012), ha="center",
                        fontsize=7, color=C_NEW)
        ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("KinFaceW-", "KFW-") for c in CORPORA],
                           rotation=20, ha="right")
        p = d["pooled"][bb]
        ax.set_title(f"{bb}   $\Delta$={p['delta']:+.3f}, $p$={p['p']:.0e}")
        ax.set_ylim(0.40, 0.90)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(frameon=False, loc="upper left", ncol=2,
               bbox_to_anchor=(0.0, 1.02), handlelength=1.2,
               columnspacing=0.8)
    save(fig, "fig_readout")


# ---------------------------------------------------------------------------
def fig_readout_sweep():
    """ROC-AUC against the number of retained observables, on FIW.

    k=12 is the full-vector run in amp_readouts.json; k<12 are the sweep runs.
    The scalar-fidelity line is the same circuit read as one number, and is the
    comparison the curve is about.
    """
    pts = []
    for k in (1, 2, 4, 6, 8, 10):
        f = os.path.join(RES, "amp_k%d.json" % k)
        if os.path.exists(f):
            j = json.load(open(f))
            if "FIW" in j:
                pts.append((k, j["FIW"]["amp-expectation"]["roc_auc_mean"]))
    full = json.load(open(os.path.join(RES, "amp_readouts.json")))["FIW"]
    pts.append((12, full["amp-expectation"]["roc_auc_mean"]))
    scalar = full["amp-fidelity"]["roc_auc_mean"]
    if len(pts) < 3:
        print("  [skip] fig_readout_sweep: too few points")
        return

    ks = [k for k, _ in pts]
    vs = [v for _, v in pts]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(ks, vs, marker="o", color=C_NEW, lw=1.5, ms=4.5,
            label="expectation, $k$ observables")
    ax.axhline(scalar, color=C_BASE, lw=1.3, ls="--",
               label="scalar fidelity (1 value)")
    ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":")
    ax.annotate("", xy=(2, vs[1]), xytext=(1, vs[0]),
                arrowprops=dict(arrowstyle="-", lw=0))
    ax.annotate("+%.3f" % (vs[1] - vs[0]), xy=(1.5, (vs[0] + vs[1]) / 2),
                xytext=(6, -2), textcoords="offset points",
                fontsize=7.5, color=C_NEW)
    # k<=6 reads person 1 only; person 2 first enters at k=8.
    ax.axvline(7, color=C_MUTED, lw=0.8, ls="--")
    ax.text(7.25, min(vs) + 0.008, "person 2 enters", fontsize=6.5,
            color=C_MUTED, rotation=90, va="bottom")
    ax.set_xlabel("observables retained, $k$")
    ax.set_ylabel("ROC-AUC (FIW)")
    ax.set_xticks(ks)
    ax.legend(frameon=False, loc="center right", fontsize=7.5,
              bbox_to_anchor=(1.0, 0.42))
    ax.set_ylim(min(min(vs), scalar) - 0.02, max(vs) + 0.012)
    save(fig, "fig_readout_sweep")


if __name__ == "__main__":
    print("Writing readout figures to paper/elsevier/figures/")
    fig_architecture()
    fig_readout()
    fig_readout_sweep()
