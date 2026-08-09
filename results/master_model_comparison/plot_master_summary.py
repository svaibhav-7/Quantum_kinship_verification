# -*- coding: utf-8 -*-
"""
Generate the final master model ranking plot consolidated from the
aggregated master comparison summary.
Writes: master_outputs/plots/00_master_ranking_4models.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_current = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(_current, "master_outputs", "master_comparison_summary.json")
out_path = os.path.join(_current, "master_outputs", "plots", "00_master_ranking_4models.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(summary_path) as f:
    summary = json.load(f)

MODEL_ORDER = ["ensemble_kinship_full", "meta_ensemble_kinship", "ensemble_kinship_fiw", "best_checkpoint"]
COLORS = {"ensemble_kinship_full": "#E91E63", "meta_ensemble_kinship": "#3F51B5",
          "ensemble_kinship_fiw": "#009688", "best_checkpoint": "#FF9800"}

models = summary["models"]
names = [models[k]["label"] for k in MODEL_ORDER]
accs = [models[k]["mean_accuracy"] if models[k]["mean_accuracy"] is not None else 0 for k in MODEL_ORDER]
aucs = [models[k]["mean_roc_auc"] * 100 if models[k]["mean_roc_auc"] is not None else 0 for k in MODEL_ORDER]
colors = [COLORS[k] for k in MODEL_ORDER]

best = summary["best_model"]
best_idx = MODEL_ORDER.index(best) if best in MODEL_ORDER else 0

x = np.arange(len(names))
w = 0.36
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - w / 2, accs, w, label="Mean Accuracy (%)", color=colors, alpha=0.9)
b2 = ax.bar(x + w / 2, aucs, w, label="Mean ROC-AUC (%)", color=colors, alpha=0.45)

for i in range(len(names)):
    ax.text(i - w / 2, accs[i] + 0.8, f"{accs[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w / 2, aucs[i] + 0.8, f"{aucs[i]:.1f}", ha="center", fontsize=9)

# Highlight best
ax.text(best_idx, max(accs + aucs) + 4, "BEST", ha="center", fontweight="bold",
        color="#3F51B5", fontsize=13, bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#3F51B5"))

ax.axvline(best_idx, color="#3F51B5", linestyle="--", alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=14, ha="right", fontsize=9)
ax.set_ylabel("Score (%)", fontweight="bold")
ax.set_title(f"Master Model Comparison - Best Model: {summary['best_model_label']}",
             fontweight="bold", fontsize=12)
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, max(accs + aucs) * 1.25)
plt.tight_layout()
plt.savefig(out_path)
plt.close()
print(f"[PLOT] saved {out_path}")
