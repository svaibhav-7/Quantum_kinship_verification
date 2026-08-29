"""Grouped k-fold evaluation for kinship datasets.

A single group-disjoint split gave accuracies spanning 61.7%-84.1% depending
on the seed, because FIW's test side was effectively two families with one
(F0282) holding 95% of the pairs. k-fold puts every group in the test set
exactly once, so no lucky draw can carry the headline number.
"""

import random
from collections import defaultdict


def grouped_kfold(pairs, keyfn, n_splits=5, seed=42):
    """Yield (train, test) with disjoint groups; every group tested once.

    Groups are dealt largest-first into the currently smallest fold, which
    keeps fold sizes even when one group is far larger than the rest.
    """
    by_group = defaultdict(list)
    for p in pairs:
        by_group[keyfn(p[0])].append(p)

    groups = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    rng = random.Random(seed)
    # Shuffle within equal-size bands so the deal is not alphabetical.
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(by_group[g]))

    folds = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for g in groups:
        i = sizes.index(min(sizes))
        folds[i].append(g)
        sizes[i] += len(by_group[g])

    for i in range(n_splits):
        test_groups = set(folds[i])
        train, test = [], []
        for p in pairs:
            ga, gb = keyfn(p[0]), keyfn(p[1])
            if ga in test_groups and gb in test_groups:
                test.append(p)
            elif ga not in test_groups and gb not in test_groups:
                train.append(p)
            # pairs bridging both sides are dropped -- they would leak
        if train and test:
            yield train, test


def balance_group_sizes(pairs, keyfn, max_per_group=2000, seed=42):
    """Cap how many pairs any one group contributes.

    FIW has 95 families but one is 19% of all pairs; without a cap it
    dominates whichever fold it lands in.
    """
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for p in pairs:
        by_group[keyfn(p[0])].append(p)

    out = []
    for g in sorted(by_group):
        items = by_group[g]
        if len(items) > max_per_group:
            items = rng.sample(items, max_per_group)
        out.extend(items)
    return out
