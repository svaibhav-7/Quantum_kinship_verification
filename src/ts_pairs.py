"""TSKinFace pair construction with shortcut-free negatives.

The stock loader builds positives entirely within a family and negatives
entirely across families (measured: 800/800 vs 0/800). A model can then score
84% by detecting a shared photo session -- lighting, camera, background --
without learning anything about kinship.

This builds a fraction of the negatives *inside* each family, pairing the
father against the mother. They are co-photographed, so they share every
session cue, but they are not parent and child. That removes 'same family' as
a usable signal while keeping the labels correct.
"""

import os
import random
from collections import defaultdict


def family_of_ts(path):
    """`.../FMS-12-F.jpg` -> `FMS-12`."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    parts = stem.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else stem


def role_of_ts(path):
    """`.../FMS-12-F.jpg` -> `F` (F/M father/mother, S/D son/daughter)."""
    return os.path.basename(path).rsplit(".", 1)[0].split("-")[-1]


def _relation(parent_role, child_role):
    if parent_role == "F":
        return "fs" if child_role == "S" else "fd"
    return "ms" if child_role == "S" else "md"


def build_tskinface_pairs(root, same_family_negative_ratio=0.5, seed=42,
                          max_families=None):
    """Return [(img1, img2, label, relation)] with honest negatives.

    same_family_negative_ratio: share of negatives drawn as father-vs-mother
    within one family. The rest stay cross-family so both regimes appear.
    """
    rng = random.Random(seed)
    families = {}

    for folder in ("FMS", "FMD"):
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            continue
        child_role = "S" if folder == "FMS" else "D"
        ids = sorted({f.rsplit(".", 1)[0].split("-")[1]
                      for f in os.listdir(fdir) if f.endswith(".jpg")
                      and len(f.rsplit(".", 1)[0].split("-")) == 3})
        if max_families:
            ids = ids[:max_families]
        for fid in ids:
            trio = {r: os.path.join(fdir, f"{folder}-{fid}-{r}.jpg")
                    for r in ("F", "M", child_role)}
            if all(os.path.exists(p) for p in trio.values()):
                families[f"{folder}-{fid}"] = (trio, child_role)

    positives = []
    for _fam, (trio, child_role) in families.items():
        for prole in ("F", "M"):
            positives.append((trio[prole], trio[child_role], 1,
                              _relation(prole, child_role)))

    n_neg = len(positives)
    n_same = int(n_neg * same_family_negative_ratio)
    fam_keys = sorted(families)

    negatives = []
    # Same-family: father vs mother -- shares the session, not the kinship.
    order = fam_keys[:]
    rng.shuffle(order)
    i = 0
    while len(negatives) < n_same and order:
        fam = order[i % len(order)]
        trio, child_role = families[fam]
        negatives.append((trio["F"], trio["M"], 0, _relation("F", child_role)))
        i += 1
        if i > len(order) * 2:
            break

    # Cross-family: parent from one family, child from another.
    while len(negatives) < n_neg and len(fam_keys) > 1:
        fa, fb = rng.sample(fam_keys, 2)
        ta, _ = families[fa]
        tb, cb = families[fb]
        prole = rng.choice(("F", "M"))
        negatives.append((ta[prole], tb[cb], 0, _relation(prole, cb)))

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs
