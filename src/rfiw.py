"""Official RFIW Track-I (kinship verification) pairs.

The official train, validation and test sides are disjoint in family and
identity, so scores here can be compared with the published FIW literature.
Splits and test labels are used exactly as distributed.

`all-pairs-split.pkl` in the same archive is deliberately not read: unpickling
a downloaded file can execute code, and the CSVs carry the same information.
"""

import csv
import glob
import os
import random

PARENT_CHILD = frozenset({"fd", "fs", "md", "ms"})


def _resolve(faces_root, relative):
    return os.path.join(faces_root, *relative.split("/"))


def load_labelled_pairs(csv_path, faces_root, relations=None):
    """(img1, img2, label, relation) rows from a labelled photo-pair CSV."""
    pairs = []
    with open(csv_path, newline="") as stream:
        for row in csv.DictReader(stream):
            relation = row["ptype"].strip().lower()
            if relations is not None and relation not in relations:
                continue
            first = _resolve(faces_root, row["p1"].strip())
            second = _resolve(faces_root, row["p2"].strip())
            if os.path.isfile(first) and os.path.isfile(second):
                pairs.append((first, second, int(row["labels"]), relation))
    return pairs


def load_identity_pairs(csv_path, faces_root, relations=None,
                        cap_per_identity_pair=None, seed=0):
    """Positive photo pairs expanded from an identity-level pair CSV."""
    rng = random.Random(seed)
    pairs = []
    with open(csv_path, newline="") as stream:
        for row in csv.DictReader(stream):
            relation = row["ptype"].strip().lower()
            if relations is not None and relation not in relations:
                continue
            first = sorted(glob.glob(
                os.path.join(_resolve(faces_root, row["p1"].strip()), "*.jpg")))
            second = sorted(glob.glob(
                os.path.join(_resolve(faces_root, row["p2"].strip()), "*.jpg")))
            expanded = [(a, b, 1, relation) for a in first for b in second]
            if cap_per_identity_pair and len(expanded) > cap_per_identity_pair:
                expanded = rng.sample(expanded, cap_per_identity_pair)
            pairs.extend(expanded)
    return pairs
