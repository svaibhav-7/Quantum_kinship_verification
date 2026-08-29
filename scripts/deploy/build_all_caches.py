# -*- coding: utf-8 -*-
"""Extract FaceNet embeddings for every dataset into one cache."""
import glob, os, pickle, sys, time
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.models_improved import FaceFeatureExtractor

OUT = os.path.join(project_root, "weights", "caches", "all_datasets_cache.pkl")

def main():
    imgs = []
    for pat in ["public/FIDs/*/*/*.jpg",
                "KinFaceW-I/**/images/**/*.jpg",
                "KinFaceW-II/**/images/**/*.jpg",
                "TSKinFace_Data/**/TSKinFace_cropped/*/*.jpg"]:
        imgs += glob.glob(os.path.join(project_root, pat), recursive=True)
    imgs = sorted(set(imgs))
    print(f"  images: {len(imgs)}")

    cache = {}
    if os.path.exists(OUT):
        cache = pickle.load(open(OUT, "rb"))
        print(f"  resuming from {len(cache)}")
    for src in ["weights/caches/fiw_full_cache.pkl"]:
        p = os.path.join(project_root, src)
        if os.path.exists(p):
            cache.update({os.path.normcase(os.path.abspath(k)): v
                          for k, v in pickle.load(open(p, "rb")).items()})

    todo = [p for p in imgs if os.path.normcase(os.path.abspath(p)) not in cache]
    print(f"  to extract: {len(todo)}")
    if todo:
        ext, t0 = FaceFeatureExtractor(), time.time()
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            try:
                for p, e in zip(batch, ext.extract_batch(batch)):
                    cache[os.path.normcase(os.path.abspath(p))] = e
            except Exception:
                for p in batch:
                    try:
                        cache[os.path.normcase(os.path.abspath(p))] = ext.extract(p)
                    except Exception as e2:
                        print(f"    skip {os.path.basename(p)}: {e2}")
            if (i // 64) % 20 == 0:
                d = min(i + 64, len(todo))
                print(f"    {d}/{len(todo)} ({d/len(todo)*100:.0f}%) "
                      f"{d/max(1e-9,time.time()-t0):.0f} img/s")
                pickle.dump(cache, open(OUT, "wb"))
    pickle.dump(cache, open(OUT, "wb"))
    print(f"  saved {len(cache)} -> {OUT}")

if __name__ == "__main__":
    main()
