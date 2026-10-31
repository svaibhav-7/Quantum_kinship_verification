"""TSKinFace pair construction and split-aware negative sampling.

Positive pairs are parent-child pairs within a family. Negative pairs combine
father-vs-mother pairs within a family and parent-child pairs across families.
The within-family negatives share photo-session cues with the positives.
"""

import os
import random
from collections import defaultdict


def family_of_ts(path):
    """Return the family key, e.g. .../FMS-12-F.jpg -> FMS-12."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    parts = stem.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else stem


def role_of_ts(path):
    """Return F/M father/mother or S/D son/daughter."""
    return os.path.basename(path).rsplit(".", 1)[0].split("-")[-1]


def _relation(parent_role, child_role):
    if parent_role == "F":
        return "fs" if child_role == "S" else "fd"
    return "ms" if child_role == "S" else "md"


def build_tskinface_pairs(root, same_family_negative_ratio=0.5, seed=42,
                          max_families=None):
    """Return (image1, image2, label, relation) pairs with balanced classes."""
    if not 0.0 <= same_family_negative_ratio <= 1.0:
        raise ValueError("same_family_negative_ratio must be between 0 and 1")

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
        if max_families is not None:
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
    if not positives:
        return []

    n_neg = len(positives)
    fam_keys = sorted(families)
    n_same_target = int(round(n_neg * same_family_negative_ratio))
    order = fam_keys[:]
    rng.shuffle(order)
    negatives = []
    for fam in order[:min(n_same_target, len(order))]:
        trio, child_role = families[fam]
        negatives.append((trio["F"], trio["M"], 0,
                          _relation("F", child_role)))

    while len(negatives) < n_neg and len(fam_keys) > 1:
        fa, fb = rng.sample(fam_keys, 2)
        ta, _ = families[fa]
        tb, cb = families[fb]
        prole = rng.choice(("F", "M"))
        negatives.append((ta[prole], tb[cb], 0, _relation(prole, cb)))

    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives + negatives


def add_tskinface_negatives(positive_pairs, keyfn=family_of_ts, seed=42,
                            same_family_negative_ratio=0.5):
    """Create negatives inside one already group-disjoint side/fold.

    Same-family father-vs-mother negatives are generated from the family-level
    parent endpoints present in the positive subset. Cross-family negatives
    sample parent and child endpoints only from families on this side.
    """
    if not 0.0 <= same_family_negative_ratio <= 1.0:
        raise ValueError("same_family_negative_ratio must be between 0 and 1")
    positives = list(positive_pairs)
    if any(int(p[2]) != 1 for p in positives):
        raise ValueError("positive_pairs must contain only positive pairs")
    if not positives:
        return []

    rng = random.Random(seed)
    by_family = defaultdict(lambda: {"parents": {}, "children": []})
    for a, b, _label, _rel in positives:
        ka, kb = keyfn(a), keyfn(b)
        if ka != kb:
            raise ValueError("TSKinFace positive pair spans multiple families")
        role = role_of_ts(a)
        by_family[ka]["parents"][role] = a
        by_family[ka]["children"].append((b, role_of_ts(b)))

    same_candidates = []
    for fam, data in by_family.items():
        parents = data["parents"]
        if "F" in parents and "M" in parents:
            child_role = data["children"][0][1]
            same_candidates.append((fam, parents["F"], parents["M"], child_role))
    rng.shuffle(same_candidates)

    target_same = int(round(len(positives) * same_family_negative_ratio))
    negatives = []
    for _fam, father, mother, child_role in same_candidates[:target_same]:
        negatives.append((father, mother, 0, _relation("F", child_role)))

    families = sorted(by_family)
    # If no cross-family choice exists, keep the available within-family
    # negatives rather than fabricating a pair across the split boundary.
    if len(families) > 1:
        remaining = max(0, len(positives) - len(negatives))
        for _ in range(remaining):
            fa = families[rng.randrange(len(families))]
            fb = rng.choice([fam for fam in families if fam != fa])
            parent_items = list(by_family[fa]["parents"].items())
            child_items = by_family[fb]["children"]
            if not parent_items or not child_items:
                continue
            parent_role, parent = rng.choice(parent_items)
            child, child_role = rng.choice(child_items)
            negatives.append((parent, child, 0,
                              _relation(parent_role, child_role)))

    return positives + negatives
