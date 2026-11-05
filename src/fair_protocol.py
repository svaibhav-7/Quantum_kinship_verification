"""Evaluation protocol v2: negatives that cannot be spotted without kinship.

Family-disjoint splitting removes identity leakage but not label leakage. The
earlier negative samplers paired a parent with a random face, so about half of
KinFaceW negatives had the wrong generation and contradicted the gender implied
by their relation label. TSKinFace added father-vs-mother negatives, always
labelled fs/fd, which made the relation label alone worth 0.62 ROC-AUC.

Here every negative keeps a real parent and substitutes a child of the same
relation from another family on the same side of the split. Generation, gender
and relation label then match the positives exactly, so they carry no signal.

This module also holds the protocol's supporting rules: balanced validation,
a validation-only failed-run rule, and family-level uncertainty.
"""

import random
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from .kfold import grouped_kfold


def _donors(positives, keyfn):
    donors = defaultdict(list)
    for _parent, child, _label, relation in positives:
        donors[relation].append((keyfn(child), child))
    return donors


def _draw(rng, candidates, family, want):
    """Up to `want` same-relation children from other families."""
    drawn = []
    for _ in range(want * 4):
        donor_family, child = candidates[rng.randrange(len(candidates))]
        if donor_family != family:
            drawn.append(child)
            if len(drawn) == want:
                return drawn
    if not drawn:  # rejection failed; check whether any donor exists at all
        drawn = [c for f, c in candidates if f != family][:want]
    return drawn


def matched_negatives(positives, keyfn, seed):
    """One negative per positive: same parent, same-relation child elsewhere.

    A relation seen in only one family has no valid donor. Those positives get
    no negative rather than a mismatched one; `build_side` then drops them.
    """
    rng = random.Random(seed)
    donors = _donors(positives, keyfn)
    negatives = []
    for parent, _child, _label, relation in positives:
        drawn = _draw(rng, donors[relation], keyfn(parent), 1)
        if drawn:
            negatives.append((parent, drawn[0], 0, relation))
    return negatives


def colour_matched_negatives(positives, keyfn, distance, seed, pool=64):
    """Matched negatives whose colour distance tracks their positive's.

    Cross-family negatives can never share a photo session, so colour and
    lighting separate them from relatives photographed together: colour alone
    scored 0.73 on TSKinFace and KinFaceW-II. Among `pool` valid donors this
    picks the child whose distance to the parent is closest to the positive's.
    Colour is only an observable proxy for a session, so this limits the cue
    without proving that it is gone.
    """
    rng = random.Random(seed)
    donors = _donors(positives, keyfn)
    negatives = []
    for parent, child, _label, relation in positives:
        drawn = _draw(rng, donors[relation], keyfn(parent), pool)
        if drawn:
            target = distance(parent, child)
            best = min(drawn, key=lambda c: abs(distance(parent, c) - target))
            negatives.append((parent, best, 0, relation))
    return negatives


def build_side(positives, keyfn, seed, distance=None):
    """Positives plus matched negatives for one side of a split.

    With `distance`, negatives are also colour-matched.
    """
    if distance is None:
        negatives = matched_negatives(positives, keyfn, seed)
    else:
        negatives = colour_matched_negatives(positives, keyfn, distance, seed)
    usable = {pair[3] for pair in negatives}
    kept = [pair for pair in positives if pair[3] in usable]
    return kept + negatives


def colour_signature(path, size=32):
    """Mean and spread of the Lab chroma channels; ignores identity and brightness."""
    from PIL import Image

    with Image.open(path) as image:
        lab = np.asarray(image.convert("LAB").resize((size, size)), dtype=np.float64)
    chroma = lab[:, :, 1:].reshape(-1, 2)
    return np.r_[chroma.mean(0), chroma.std(0)]


class ColourDistance:
    """Memoised colour distance between image paths."""

    def __init__(self):
        self._memo = {}

    def signature(self, path):
        if path not in self._memo:
            self._memo[path] = colour_signature(path)
        return self._memo[path]

    def __call__(self, first, second):
        return float(np.linalg.norm(self.signature(first) - self.signature(second)))


def split_fit_val(positives, keyfn, seed, n_splits=5):
    """Group-disjoint fit/validation split, taken on positives only.

    Splitting before negatives exist is what keeps validation balanced: the old
    pooled slice discarded any cross-family negative with an endpoint outside
    validation, leaving it about 84% positive.
    """
    if any(int(pair[2]) != 1 for pair in positives):
        raise ValueError("split_fit_val takes positive pairs only")
    fit, val = next(iter(grouped_kfold(positives, keyfn, n_splits, seed)))
    return fit, val


def drop_duplicate_groups(pairs, keyfn, embed, decimals=4):
    """Remove groups whose whole image set duplicates an earlier group.

    Returns (kept_pairs, dropped_group_keys). A single shared face is not
    enough: only a family stored twice crosses folds as a true duplicate.
    """
    images = defaultdict(set)
    for first, second, _label, _relation in pairs:
        images[keyfn(first)].add(first)
        images[keyfn(second)].add(second)

    seen, dropped = {}, []
    for group in sorted(images):
        signature = []
        for path in images[group]:
            vector = np.asarray(embed(path), dtype=np.float64)
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            signature.append(tuple(np.round(vector, decimals).tolist()))
        key = tuple(sorted(signature))
        if key in seen:
            dropped.append(group)
        else:
            seen[key] = group

    removed = set(dropped)
    kept = [p for p in pairs if keyfn(p[0]) not in removed]
    return kept, dropped


def is_failed_run(val_auc, val_cosine_auc, slack=0.02):
    """True when a trained model cannot match untrained cosine on validation.

    The rule reads validation only, so applying it cannot select on test.
    """
    return val_auc < val_cosine_auc - slack


def t_interval(values, confidence=0.95):
    """(mean, half-width) from Student's t; half-width is NaN for one value."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan")
    se = values.std(ddof=1) / np.sqrt(len(values))
    critical = stats.t.ppf(0.5 + confidence / 2.0, len(values) - 1)
    return mean, float(critical * se)


def bootstrap_auc_delta(y, scores_a, scores_b, groups_a, groups_b=None,
                        n_boot=2000, seed=0):
    """ROC-AUC difference with a two-endpoint family-cluster percentile CI.

    Cross-family pairs are dyads: resampling only one endpoint misses
    dependence through the donor family. This uses a two-way pigeonhole
    bootstrap, weighting each cross-family pair by the product of its two
    family multiplicities and each within-family pair by that family's
    multiplicity. It is an interval procedure, not a significance test.
    """
    y = np.asarray(y)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    groups_a = np.asarray(groups_a)
    if groups_b is None:
        groups_b = groups_a
    groups_b = np.asarray(groups_b)
    if not (y.ndim == scores_a.ndim == scores_b.ndim == 1
            and groups_a.ndim == groups_b.ndim == 1
            and len(y) == len(scores_a) == len(scores_b)
            == len(groups_a) == len(groups_b)):
        raise ValueError("labels, scores and both endpoint-group arrays must align")
    if len(y) == 0 or len(np.unique(y)) != 2:
        raise ValueError("AUC bootstrap requires non-empty labels from both classes")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    keys = np.unique(np.concatenate((groups_a, groups_b)))
    if len(keys) < 2:
        raise ValueError("AUC bootstrap requires at least two family clusters")
    lookup = {group: index for index, group in enumerate(keys.tolist())}
    index_a = np.asarray([lookup[group] for group in groups_a], dtype=np.int64)
    index_b = np.asarray([lookup[group] for group in groups_b], dtype=np.int64)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        counts = np.bincount(
            rng.integers(0, len(keys), len(keys)), minlength=len(keys)
        )
        weights = counts[index_a] * counts[index_b]
        same_family = index_a == index_b
        weights[same_family] = counts[index_a[same_family]]
        selected = weights > 0
        if len(np.unique(y[selected])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[selected], scores_a[selected],
                          sample_weight=weights[selected])
            - roc_auc_score(y[selected], scores_b[selected],
                            sample_weight=weights[selected])
        )
    if len(deltas) < max(100, n_boot // 2):
        raise RuntimeError(
            f"Only {len(deltas)}/{n_boot} valid family-bootstrap samples"
        )

    low, high = np.percentile(deltas, [2.5, 97.5])
    deltas = np.asarray(deltas)
    return {
        "delta": float(roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_groups": len(keys),
        "n_boot": len(deltas),
        "bootstrap": "two-way family pigeonhole; paired score difference",
    }


def adjust_pvalues(pvalues, method):
    """Holm ("holm") or Benjamini-Hochberg ("bh") adjustment of one test family."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=np.float64)
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (n - rank) * pvalues[index])
            adjusted[index] = min(1.0, running)
    elif method == "bh":
        running = 1.0
        for rank in range(n - 1, -1, -1):
            index = order[rank]
            running = min(running, pvalues[index] * n / (rank + 1))
            adjusted[index] = running
    else:
        raise ValueError(f"unknown method {method!r}")
    return adjusted.tolist()


def pool_fold_scores(folds):
    """Pool within-fold ranks and retain both family IDs of every dyad.

    Each fold's model has its own score scale, so scores are ranked within a
    fold and seed, then averaged across seeds. `folds` holds dicts with
    "scores" (one array per seed), "y", "groups_a" and "groups_b".
    Returns (y, scores, groups_a, groups_b).
    """
    ys, pooled, groups_a, groups_b = [], [], [], []
    for fold in folds:
        ranks = [stats.rankdata(values) / len(values) for values in fold["scores"]]
        pooled.append(np.mean(ranks, axis=0))
        ys.append(fold["y"])
        groups_a.append(fold["groups_a"])
        groups_b.append(fold["groups_b"])
    return (np.concatenate(ys), np.concatenate(pooled),
            np.concatenate(groups_a), np.concatenate(groups_b))
