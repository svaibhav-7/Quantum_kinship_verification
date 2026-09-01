# -*- coding: utf-8 -*-
"""Extract ArcFace embeddings for every benchmark image.

A second backbone answers a question the paper cannot otherwise settle: are
the leakage and person-level findings properties of the task and its
benchmarks, or artefacts of FaceNet? ArcFace (buffalo_l, w600k_r50) is trained
with a different objective on a different corpus, so agreement across the two
is meaningful evidence.

Benchmark images are already tight face crops, so detection is bypassed and the
recognition model is applied directly -- matching how the FaceNet cache was
built, which keeps the comparison like-for-like.
"""
import glob
import os
import pickle
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

OUT = os.path.join(project_root, "weights", "caches", "arcface_cache.pkl")


def image_list():
    pats = ["public/FIDs/*/*/*.jpg",
            "KinFaceW-I/**/images/**/*.jpg",
            "KinFaceW-II/**/images/**/*.jpg",
            "TSKinFace_Data/**/TSKinFace_cropped/*/*.jpg"]
    out = []
    for p in pats:
        out += glob.glob(os.path.join(project_root, p), recursive=True)
    return sorted(set(out))


def main():
    import cv2
    from insightface.model_zoo import get_model

    imgs = image_list()
    print(f"  images: {len(imgs)}")

    cache = {}
    if os.path.exists(OUT):
        cache = pickle.load(open(OUT, "rb"))
        print(f"  resuming from {len(cache)}")

    todo = [p for p in imgs if os.path.normcase(os.path.abspath(p)) not in cache]
    print(f"  to extract: {len(todo)}")
    if not todo:
        print("  nothing to do")
        return

    model_path = os.path.join(os.path.expanduser("~"), ".insightface", "models",
                              "buffalo_l", "w600k_r50.onnx")
    if not os.path.exists(model_path):
        sys.exit(f"ArcFace weights not found at {model_path}\n"
                 f"Run once: python -c \"from insightface.app import FaceAnalysis; "
                 f"FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\"")

    rec = get_model(model_path)
    rec.prepare(ctx_id=-1)
    print("  ArcFace w600k_r50 loaded")

    t0 = time.time()
    for i, p in enumerate(todo):
        try:
            img = cv2.imread(p)
            if img is None:
                print(f"    skip unreadable: {os.path.basename(p)}")
                continue
            # Benchmark crops are already aligned; ArcFace expects 112x112.
            img = cv2.resize(img, (112, 112))
            v = rec.get_feat(img).ravel()
            cache[os.path.normcase(os.path.abspath(p))] = (
                v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
        except Exception as e:
            print(f"    skip {os.path.basename(p)}: {e}")

        if i and i % 500 == 0:
            print(f"    {i}/{len(todo)} ({i/len(todo)*100:.0f}%) "
                  f"{i/max(1e-9, time.time()-t0):.0f} img/s", flush=True)
            pickle.dump(cache, open(OUT, "wb"))

    pickle.dump(cache, open(OUT, "wb"))
    print(f"  saved {len(cache)} embeddings -> {os.path.relpath(OUT, project_root)}")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
