"""Load all four kinship datasets with honest, leakage-free splits.

Each dataset is split on its own family/identity unit before pooling, so a
family never spans train and test regardless of which dataset it came from.
"""

import os

from .data_loaders import load_kinfacew_pairs, load_fiw_pairs
from .splits import split_then_build, family_of
from .ts_pairs import build_tskinface_pairs, family_of_ts


def _kfw_family(path):
    """KinFaceW filenames look like `fd_001_1.jpg`; the family is `fd_001`."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def _split_by_key(pairs, keyfn, test_ratio, seed):
    """Group-disjoint split on an arbitrary key (used for KinFaceW/TSKinFace,
    whose negatives are already fixed and cannot be regenerated per side)."""
    import random

    groups = sorted({keyfn(p[0]) for p in pairs} | {keyfn(p[1]) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_ratio))
    test_groups = set(groups[:n_test])

    train, test = [], []
    for p in pairs:
        ga, gb = keyfn(p[0]), keyfn(p[1])
        if ga in test_groups and gb in test_groups:
            test.append(p)
        elif ga not in test_groups and gb not in test_groups:
            train.append(p)
        # pairs bridging the two sides are dropped -- they would leak
    return train, test


def split_rebuild_negatives(pairs, keyfn, test_ratio, seed):
    """Split on positives only, then rebuild negatives inside each side.

    Fixed negatives that join two groups straddle any group-disjoint boundary
    and must be discarded -- on KinFaceW-I that lost 173/1066 pairs and left
    only 119 for test. Rebuilding them per side keeps the data and the
    label semantics (a negative is still two different families).
    """
    import random

    rng = random.Random(seed)
    positives = [p for p in pairs if p[2] == 1]
    groups = sorted({keyfn(p[0]) for p in positives} | {keyfn(p[1]) for p in positives})
    rng.shuffle(groups)
    test_g = set(groups[:max(1, int(len(groups) * test_ratio))])

    sides = ([], [])
    for p in positives:
        ga, gb = keyfn(p[0]), keyfn(p[1])
        if ga in test_g and gb in test_g:
            sides[1].append(p)
        elif ga not in test_g and gb not in test_g:
            sides[0].append(p)

    out = []
    for si, pos in enumerate(sides):
        by_group = {}
        for a, b, _l, _r in pos:
            by_group.setdefault(keyfn(a), []).append(a)
            by_group.setdefault(keyfn(b), []).append(b)
        keys = sorted(by_group)
        neg = []
        if len(keys) > 1:
            r2 = random.Random(seed + 100 + si)
            for a, _b, _l, rel in pos:
                ka = keyfn(a)
                others = [k for k in keys if k != ka]
                kb = others[r2.randrange(len(others))]
                pool = by_group[kb]
                neg.append((a, pool[r2.randrange(len(pool))], 0, rel))
        out.append(pos + neg)
    return out[0], out[1]


def load_all_datasets(project_root, test_ratio=0.2, seed=42,
                      ts_negative_ratio=0.5):
    """Return (train_pairs, test_pairs, per_dataset_test) across all datasets."""
    train, test, per_ds = [], [], {}

    kin = [p for p in load_fiw_pairs(os.path.join(project_root, "public"))
           if p[2] == 1]
    tr, te = split_then_build(kin, test_ratio=test_ratio, seed=seed)
    train += tr; test += te; per_ds["FIW"] = te

    for folder in ("KinFaceW-I", "KinFaceW-II"):
        root = os.path.join(project_root, folder)
        if not os.path.exists(root):
            continue
        raw = load_kinfacew_pairs(root)
        ps = [(r[0], r[1], r[2], r[3]) for r in raw]
        tr, te = split_rebuild_negatives(ps, _kfw_family, test_ratio, seed)
        train += tr; test += te; per_ds[folder] = te

    ts_root = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                           "TSKinFace_cropped")
    if os.path.exists(ts_root):
        ps = build_tskinface_pairs(ts_root,
                                   same_family_negative_ratio=ts_negative_ratio,
                                   seed=seed)
        tr, te = split_rebuild_negatives(ps, family_of_ts, test_ratio, seed)
        # split_rebuild_negatives regenerates cross-family negatives only, so
        # re-attach the same-family (father-vs-mother) ones that close the
        # photo-session shortcut. They live entirely inside one family, so
        # they never cross the split boundary.
        sf = [p for p in ps if p[2] == 0 and family_of_ts(p[0]) == family_of_ts(p[1])]
        te_g = {family_of_ts(p[0]) for p in te}
        tr += [p for p in sf if family_of_ts(p[0]) not in te_g]
        te += [p for p in sf if family_of_ts(p[0]) in te_g]
        train += tr; test += te; per_ds["TSKinFace"] = te

    return train, test, per_ds


def dataset_group_key(path):
    """Grouping unit scoped to its dataset.

    `fd_003` names a different family in KinFaceW-I than in KinFaceW-II, so an
    unscoped key merges two unrelated families and corrupts any pooled
    disjointness check. Prefixing with the dataset keeps them apart.
    """
    from .splits import family_of
    from .ts_pairs import family_of_ts

    norm = path.replace("\\", "/")
    if "FIDs" in norm:
        return "fiw:" + family_of(path)
    if "TSKinFace" in norm:
        return "ts:" + family_of_ts(path)
    if "KinFaceW-II" in norm:
        return "kfw2:" + _kfw_family(path)
    return "kfw1:" + _kfw_family(path)
