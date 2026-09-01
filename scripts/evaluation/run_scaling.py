# -*- coding: utf-8 -*-
"""Cost of person-level scoring against photo-set size.

Person-level aggregation compares two photo sets, so the cross-set similarity
matrix it depends on is quadratic in set size. That bounds how the method
deploys, and the bound has a convenient interaction with the accuracy curve:
the gain saturates around five photographs, which is also where the cost is
still negligible.

Writes results/honest/scaling_curve.json.
"""
import json
import os
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.identity_sets import pair_set_features

SIZES = [1, 2, 5, 10, 20, 40, 80, 160, 320]
DIM = 512
REPEATS = 400
TRIALS = 5  # median of several timings; single timings were unstable


def main():
    rng = np.random.default_rng(0)
    rows = []
    print(f"  {'photos':>7s} {'ms/pair':>10s} {'pairs/s':>10s} {'vs n=1':>9s}")
    print("  " + "-" * 40)

    base = None
    for n in SIZES:
        A = rng.normal(size=(n, DIM))
        B = rng.normal(size=(n, DIM))
        pair_set_features(A, B)  # warm up

        reps = max(20, REPEATS // max(1, n // 5))
        # Median of several trials: single timings varied by >10x at large n,
        # which produced a spurious super-quadratic exponent.
        trials = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            for _ in range(reps):
                pair_set_features(A, B)
            trials.append((time.perf_counter() - t0) / reps)
        dt = float(np.median(trials))

        rate = 1.0 / dt
        if base is None:
            base = rate
        rows.append({"photos": n, "ms_per_pair": dt * 1000.0,
                     "pairs_per_sec": rate, "slowdown_vs_1": base / rate})
        print(f"  {n:7d} {dt*1000:10.3f} {rate:10.0f} {base/rate:8.1f}x")

    # Confirm the growth is quadratic rather than merely superlinear.
    big = [r for r in rows if r["photos"] >= 25]
    if len(big) >= 2:
        lo, hi = big[0], big[-1]
        ratio_n = hi["photos"] / lo["photos"]
        ratio_t = hi["ms_per_pair"] / lo["ms_per_pair"]
        expo = np.log(ratio_t) / np.log(ratio_n)
        print(f"\n  empirical exponent over n=[{lo['photos']},{hi['photos']}]: "
              f"{expo:.2f}  (2.0 = quadratic)")
    else:
        expo = float("nan")

    out = {"dim": DIM, "curve": rows, "empirical_exponent": float(expo)}
    path = os.path.join(project_root, "results", "honest", "scaling_curve.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n  saved results/honest/scaling_curve.json")


if __name__ == "__main__":
    main()
