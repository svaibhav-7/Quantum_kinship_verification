"""Family-disjoint splitting for kinship datasets.

The previous training code split on pair index. Because FIW reuses the same
472 identities across 57k pairs, that put an identity from every single test
pair somewhere in the training set (measured: 11424/11424 = 100%). Every
metric produced that way is optimistic.

Splitting on the family is the only split that holds: a family is the unit
that shares identities, photo sessions, and capture conditions.
"""

import os
from collections import defaultdict


def family_of(path):
    """`.../FIDs/F0282/MID3/P01.jpg` -> `F0282`."""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def identity_of(path):
    """`.../FIDs/F0282/MID3/P01.jpg` -> `F0282/MID3`."""
    return family_of(path) + "/" + os.path.basename(os.path.dirname(path))


def _pair_families(pair):
    return {family_of(pair[0]), family_of(pair[1])}


def family_disjoint_split(pairs, test_ratio=0.2, seed=42):
    """Split pairs so no family — hence no identity — spans both sides.

    Non-kin pairs join two families, which transitively links them: if F1 is
    paired with F2, both must land on the same side. We resolve those links
    into connected components with union-find and assign whole components,
    largest first, into whichever side is furthest from its target size.
    Greedy-largest-first matters because FIW is extremely unbalanced (one
    family is 19% of all pairs), so a random assignment overshoots badly.

    Returns (train_pairs, test_pairs).
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        fams = list(_pair_families(p))
        for f in fams:
            find(f)
        for f in fams[1:]:
            union(fams[0], f)

    # Group pairs by the component they belong to.
    comp_pairs = defaultdict(list)
    for p in pairs:
        comp_pairs[find(family_of(p[0]))].append(p)

    target_test = len(pairs) * test_ratio
    # Deterministic order: size desc, then component key for tie-breaking.
    ordered = sorted(comp_pairs.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    train, test = [], []
    for _, group in ordered:
        # Send the group wherever it leaves the test side closest to target.
        if len(test) + len(group) <= target_test or not test:
            test.extend(group)
        else:
            train.extend(group)

    # Guarantee both sides non-empty even in degenerate single-component input.
    if not train:
        train, test = test, train
        if not test and len(train) > 1:
            test = [train.pop()]
    if not test and len(train) > 1:
        test = [train.pop()]

    return train, test


def summarize_split(train, test):
    """Leakage report for a split. Returns a dict; all counts should be 0."""
    tr_f = {family_of(p[0]) for p in train} | {family_of(p[1]) for p in train}
    te_f = {family_of(p[0]) for p in test} | {family_of(p[1]) for p in test}
    tr_i = {identity_of(p[0]) for p in train} | {identity_of(p[1]) for p in train}
    te_i = {identity_of(p[0]) for p in test} | {identity_of(p[1]) for p in test}
    return {
        "n_train": len(train),
        "n_test": len(test),
        "test_fraction": len(test) / max(1, len(train) + len(test)),
        "shared_families": len(tr_f & te_f),
        "shared_identities": len(tr_i & te_i),
        "train_kin_ratio": sum(p[2] for p in train) / max(1, len(train)),
        "test_kin_ratio": sum(p[2] for p in test) / max(1, len(test)),
    }


def build_negatives(kin_pairs, seed=42):
    """Generate one cross-family negative per kin pair, drawn only from the
    families present in `kin_pairs`. Keeping generation inside one side is
    what allows the split to stay disjoint."""
    import random

    rng = random.Random(seed)
    by_family = defaultdict(list)
    for a, b, _, _ in kin_pairs:
        by_family[family_of(a)].append(a)
        by_family[family_of(b)].append(b)

    families = sorted(by_family)
    if len(families) < 2:
        return []

    negatives = []
    for a, _b, _label, rel in kin_pairs:
        fa = family_of(a)
        others = [f for f in families if f != fa]
        if not others:
            continue
        of = others[rng.randrange(len(others))]
        pool = by_family[of]
        negatives.append((a, pool[rng.randrange(len(pool))], 0, rel))
    return negatives


def split_then_build(kin_pairs, test_ratio=0.2, seed=42):
    """Split on family using positives only, then build negatives per side.

    Negatives cannot exist before the split: each one bridges two families,
    and on real FIW that collapses all 95 families into a single connected
    component, making a disjoint split impossible. Positives alone keep
    families separable.

    Returns (train_pairs, test_pairs), each 1:1 balanced.
    """
    positives = [p for p in kin_pairs if p[2] == 1]
    train_pos, test_pos = family_disjoint_split(
        positives, test_ratio=test_ratio, seed=seed
    )
    train = train_pos + build_negatives(train_pos, seed=seed)
    test = test_pos + build_negatives(test_pos, seed=seed + 1)
    return train, test
