# -*- coding: utf-8 -*-
"""
MODULE 08: COMPUTATIONAL EFFICIENCY (per model)

For each of the 4 models, measures parameter count, memory footprint,
single-pair latency, batch-128 latency, FPS, and throughput.
Produces per-model JSON + a comparison bar chart.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_current = os.path.dirname(os.path.abspath(__file__))
_module_root = os.path.dirname(_current)
if _module_root not in sys.path:
    sys.path.insert(0, _module_root)

from common_models import get_models, MODEL_KEYS, MODEL_LABELS
from common_utils import save_json

OUT_DIR = os.path.join(_module_root, "08_computational_efficiency")
os.makedirs(OUT_DIR, exist_ok=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_efficiency():
    print("\n" + "=" * 70)
    print("  MODULE 08: COMPUTATIONAL EFFICIENCY (4 MODELS)")
    print("=" * 70)

    models = get_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}

    for mkey in MODEL_KEYS:
        model = models[mkey].to(device).eval()
        n_params = count_params(model)
        mem_mb = (n_params * 4) / (1024 * 1024)

        e1 = torch.randn(1, 512, device=device)
        e2 = torch.randn(1, 512, device=device)
        rel = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(e1, e2, rel)

        # Single-pair latency
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                model(e1, e2, rel)
        t1 = time.perf_counter()
        single_ms = ((t1 - t0) / 100) * 1000
        single_fps = 1000.0 / single_ms

        # Batch-128 latency
        e1b = torch.randn(128, 512, device=device)
        e2b = torch.randn(128, 512, device=device)
        relb = torch.tile(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device), (128, 1))
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(30):
                model(e1b, e2b, relb)
        t1 = time.perf_counter()
        batch_total = ((t1 - t0) / 30) * 1000
        batch_per_pair = batch_total / 128
        batch_fps = 1000.0 / batch_per_pair

        res = {
            "parameters": int(n_params),
            "memory_footprint_mb": float(round(mem_mb, 2)),
            "single_pair_latency_ms": float(round(single_ms, 2)),
            "single_pair_fps": float(round(single_fps, 1)),
            "batch128_total_ms": float(round(batch_total, 2)),
            "batch128_per_pair_ms": float(round(batch_per_pair, 2)),
            "batch128_fps": float(round(batch_fps, 1)),
        }
        save_json(res, os.path.join(OUT_DIR, f"{mkey}.json"))
        all_results[mkey] = res
        print(f"  [MODEL {mkey}] params={n_params}, fsingle={single_fps:.1f}, fbatch={batch_fps:.1f}")

    _plot_combined(all_results)
    save_json(all_results, os.path.join(OUT_DIR, "efficiency_summary.json"))
    print("  [MODULE 08 COMPLETE]")
    return all_results


def _plot_combined(all_results):
    names = [MODEL_LABELS[k] for k in MODEL_KEYS]
    params = [all_results[k]["parameters"] for k in MODEL_KEYS]
    fps = [all_results[k]["single_pair_fps"] for k in MODEL_KEYS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(names, params, color=["#E91E63", "#3F51B5", "#009688", "#FF9800"])
    ax1.set_ylabel("Parameters", fontweight="bold")
    ax1.set_title("Model Size (Parameters)", fontweight="bold")
    ax1.set_xticklabels(names, rotation=12, ha="right", fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2.bar(names, fps, color=["#E91E63", "#3F51B5", "#009688", "#FF9800"])
    ax2.set_ylabel("Single-Pair FPS", fontweight="bold")
    ax2.set_title("Single-Pair Throughput (FPS)", fontweight="bold")
    ax2.set_xticklabels(names, rotation=12, ha="right", fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "efficiency_combined_4models.png"))
    plt.close()


if __name__ == "__main__":
    run_efficiency()
