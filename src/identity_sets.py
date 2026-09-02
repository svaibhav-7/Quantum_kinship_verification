"""Identity photo sets and set-level pair features.

The pairwise pipeline scored one photograph against one photograph. FIW carries
a median of five photographs per identity, so most of the available evidence was
discarded. Representing each person by their photo set lifts raw separability
from 0.734 to 0.805 ROC-AUC with no training -- a larger gain than any
architectural change measured on this project.

Degradation is the load-bearing property here. A set of size one must reduce
exactly to the single-image path so that KinFaceW, which has one photograph per
person, is untouched. `test_identity_sets.py` asserts this rather than trusting
it.
"""

import numpy as np

from .splits import family_of


def identity_of_path(path):
    """Stable per-person key, whichever dataset the path comes from.

    FIW nests photos under FIDs/<family>/MID<n>/, so the directory identifies
    the person and grouping is meaningful. KinFaceW and TSKinFace store one
    photo per person in a flat directory, where the directory identifies the
    *relation*, not the individual -- keying on it collapsed all 533 KinFaceW-I
    people into 4 buckets and fabricated set sizes of 55-169 for a corpus with
    exactly one photo each. Those datasets therefore key on the file itself,
    which makes every set a singleton and holds the degradation guarantee.
    """
    import os

    norm = path.replace("\\", "/")
    if "/FIDs/" in norm or "/MID" in norm:
        return family_of(path) + "/" + os.path.basename(os.path.dirname(path))
    return os.path.basename(norm).rsplit(".", 1)[0]


def build_identity_sets(pairs):
    """Map each identity to the list of distinct image paths depicting it."""
    import os

    seen, sets = {}, {}
    for p in pairs:
        for path in (p[0], p[1]):
            ident = identity_of_path(path)
            key = os.path.normcase(os.path.abspath(path))
            bucket = seen.setdefault(ident, set())
            if key not in bucket:
                bucket.add(key)
                sets.setdefault(ident, []).append(path)
    return sets


def _unit(X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def set_descriptor(X):
    """Summarise one identity's photo set.

    mean   -- L2-normalised centroid; carries most of the signal
    spread -- 1 - mean pairwise cosine; how variable the person looks.
              Exactly 0 for a single image, which is what makes the
              single-image path a strict special case.
    n      -- set size, so a model can learn to discount small sets
    """
    Xn = _unit(X)
    m = Xn.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)

    n = Xn.shape[0]
    if n < 2:
        spread = 0.0
    else:
        # Closed form for the mean off-diagonal Gram entry. For unit rows,
        #     sum_{i != j} <x_i, x_j> = ||sum_i x_i||^2 - n
        # so the pairwise mean needs no n x n matrix and no index arrays.
        # The explicit version built G and called triu_indices, which
        # allocated ~n^2/2 index pairs and cost 46 ms of a 48 ms call at
        # n = 160. Equivalence is pinned by tests.
        total = float(Xn.sum(axis=0) @ Xn.sum(axis=0)) - n
        spread = 1.0 - total / (n * (n - 1))
    return {"mean": m, "spread": max(0.0, spread), "n": n}


def _density(Xn, eps=1e-3):
    """rho = (1/N) sum |psi_i><psi_i|, trace-normalised."""
    R = (Xn[:, :, None] * Xn[:, None, :]).mean(axis=0)
    R = R + eps * np.eye(R.shape[0]) / R.shape[0]
    return R / np.trace(R)


DENSITY_DIM = 32
"""Subspace size for the density arm.

rho is d x d estimated from a median of 5 images. At d=512 it is hopelessly
rank-deficient and costs 512^2 per identity; projecting to 32 dimensions gives
the formulation its best available chance while keeping the cost tractable.
Even so it measures 0.7185 against mean-pooling's 0.8050.
"""


def pair_set_features(A, B, with_density=False):
    """Features for a pair of identity photo sets.

    Cross-set statistics (max/min/mean/std of the similarity matrix) are the
    ones that measured 0.8113 in probe 2, above mean-pooling's 0.8050. All of
    them collapse to the single cosine when both sets are singletons.

    `with_density` adds Tr(rho_a rho_b), the density-matrix fidelity. It is a
    pre-registered control and defaults OFF: measured in isolation it scores
    0.7185 against mean-pooling's 0.8050, and adds +0.001 when combined. It is
    retained so the claim stays falsifiable, not because it helps.
    """
    An, Bn = _unit(A), _unit(B)
    da, db = set_descriptor(An), set_descriptor(Bn)

    S = An @ Bn.T
    feats = {
        "cos_mean": float(da["mean"] @ db["mean"]),
        "x_max": float(S.max()),
        "x_min": float(S.min()),
        "x_mean": float(S.mean()),
        "x_std": float(S.std()),
        "spread_a": da["spread"],
        "spread_b": db["spread"],
        "n_a": float(da["n"]),
        "n_b": float(db["n"]),
    }

    if with_density:
        d = min(DENSITY_DIM, An.shape[1])
        Ra, Rb = _density(An[:, :d]), _density(Bn[:, :d])
        feats["density_fid"] = float(np.clip(np.sum(Ra * Rb.T), 0.0, 1.0))
    return feats


FEATURE_ORDER = ("cos_mean", "x_max", "x_min", "x_mean", "x_std",
                 "spread_a", "spread_b", "n_a", "n_b")


def features_to_vector(feats, with_density=False):
    """Stable ordering so a trained model and inference agree."""
    keys = list(FEATURE_ORDER)
    if with_density:
        keys.append("density_fid")
    return np.array([feats[k] for k in keys], dtype=np.float32)
