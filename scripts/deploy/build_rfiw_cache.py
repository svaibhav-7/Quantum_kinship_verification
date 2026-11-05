# -*- coding: utf-8 -*-
"""Extract the official RFIW Track-I faces and embed them with FaceNet.

Faces go to public/rfiw_track1/ and embeddings to weights/caches/, both
git-ignored. The run resumes from an existing cache.
"""
import csv
import glob
import os
import pickle
import re
import sys
import tarfile
import time
import zipfile

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models_improved import FaceFeatureExtractor

ARCHIVE = os.path.join(project_root, "public", "track-I.zip")
FIDS_ARCHIVE = os.path.join(project_root, "public", "FIDs.zip")
ROOT = os.path.join(project_root, "public", "rfiw_track1")
OUT = os.path.join(project_root, "weights", "caches", "rfiw_track1_cache.pkl")
MEMBERS = ("train-faces", "val-faces", "test-faces")
CSVS = ("train-pairs.csv", "val-pairs.csv", "test-pairs.csv")


def extract():
    os.makedirs(ROOT, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in CSVS:
            target = os.path.join(ROOT, name)
            if not os.path.exists(target):
                with open(target, "wb") as stream:
                    stream.write(archive.read("track-I/" + name))
        for member in MEMBERS:
            if os.path.isdir(os.path.join(ROOT, member)):
                continue
            print("  extracting", member, flush=True)
            with archive.open("track-I/%s.tar.xz" % member) as raw:
                with tarfile.open(fileobj=raw, mode="r:xz") as tar:
                    tar.extractall(ROOT, filter="data")


def extract_test_faces():
    """Test pairs name faces by family path, but test-faces.tar.xz is anonymised
    (face0.jpg, ...). The same files carry those paths in FIDs.zip."""
    target_root = os.path.join(ROOT, "test-fids")
    with open(os.path.join(ROOT, "test-pairs.csv"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    wanted = {row[column].strip() for row in rows for column in ("p1", "p2")}
    missing = [w for w in wanted
               if not os.path.exists(os.path.join(target_root, *w.split("/")))]
    if not missing:
        return
    valid = re.compile(r"^F\d+/MID\d+/[\w.-]+\.jpg$")
    with zipfile.ZipFile(FIDS_ARCHIVE) as archive:
        names = set(archive.namelist())
        for relative in missing:
            member = "FIDs/" + relative
            if not valid.match(relative) or member not in names:
                continue
            target = os.path.join(target_root, *relative.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as stream:
                stream.write(archive.read(member))


def main():
    extract()
    extract_test_faces()
    images = sorted(
        glob.glob(os.path.join(ROOT, "*-faces", "*", "*", "*.jpg"))
        + glob.glob(os.path.join(ROOT, "test-fids", "*", "*", "*.jpg")))
    print("  images:", len(images), flush=True)
    cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    key = lambda path: os.path.normcase(os.path.abspath(path))
    todo = [path for path in images if key(path) not in cache]
    print("  to embed:", len(todo), flush=True)
    if todo:
        extractor, started = FaceFeatureExtractor(), time.time()
        for start in range(0, len(todo), 128):
            batch = todo[start:start + 128]
            for path, embedding in zip(batch, extractor.extract_batch(batch)):
                cache[key(path)] = embedding
            if (start // 128) % 50 == 0:
                done = start + len(batch)
                print("    %d/%d %.0f img/s" % (
                    done, len(todo), done / max(1e-9, time.time() - started)),
                    flush=True)
                pickle.dump(cache, open(OUT, "wb"))
    pickle.dump(cache, open(OUT, "wb"))
    print("  saved", len(cache), "->", OUT, flush=True)


if __name__ == "__main__":
    main()
