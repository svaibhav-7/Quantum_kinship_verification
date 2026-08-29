# -*- coding: utf-8 -*-
"""Extract FaceNet embeddings for every FIW image into one clean cache.

The existing caches hold stale absolute paths from a previous D:\ location,
so only half their entries resolve. This writes a single cache keyed by
normalised absolute paths for the images actually present.
"""
import os, pickle, sys, time
import numpy as np, torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.models_improved import FaceFeatureExtractor

OUT = os.path.join(project_root, "weights", "caches", "fiw_full_cache.pkl")


def main():
    import glob
    imgs = sorted(glob.glob(os.path.join(project_root, "public", "FIDs", "*", "*", "*.jpg")))
    print(f"  images found: {len(imgs)}")

    cache = {}
    if os.path.exists(OUT):
        with open(OUT, "rb") as f:
            cache = pickle.load(f)
        print(f"  resuming from {len(cache)} cached")

    todo = [p for p in imgs if os.path.normcase(os.path.abspath(p)) not in cache]
    print(f"  to extract: {len(todo)}")
    if not todo:
        print("  nothing to do"); return

    ext = FaceFeatureExtractor()
    bs, t0 = 64, time.time()
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        try:
            embs = ext.extract_batch(batch)
            for p, e in zip(batch, embs):
                cache[os.path.normcase(os.path.abspath(p))] = e
        except Exception as ex:
            # Fall back to one-at-a-time so a single bad file cannot kill the run.
            for p in batch:
                try:
                    cache[os.path.normcase(os.path.abspath(p))] = ext.extract(p)
                except Exception as e2:
                    print(f"    skip {os.path.basename(p)}: {e2}")
        if (i // bs) % 10 == 0:
            done = min(i + bs, len(todo))
            print(f"    {done}/{len(todo)}  ({done/len(todo)*100:.0f}%)  "
                  f"{done/max(1e-9, time.time()-t0):.0f} img/s")
            with open(OUT, "wb") as f:
                pickle.dump(cache, f)

    with open(OUT, "wb") as f:
        pickle.dump(cache, f)
    print(f"  saved {len(cache)} embeddings -> {OUT}")


if __name__ == "__main__":
    main()
