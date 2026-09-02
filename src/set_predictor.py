"""Deployment inference for the set-level and triadic models.

These are the two components that measured a real gain under grouped k-fold:

  set-level  FIW 0.7296 -> 0.8066 (+0.077); exactly 0.000 on the three
             single-photograph corpora, by construction
  triadic    TSKinFace 0.7569 -> 0.7893 (+0.0324, p = 0.0001)

Both operate on features rather than raw embeddings, so the fitted model is a
scaler plus a logistic regression -- small enough to ship as a single file and
cheap enough to score in microseconds.

Supplying one photograph per person is fully supported: the set descriptors
collapse to the single-image case, which is why single-photograph datasets are
unchanged.
"""

import os

import numpy as np

from .identity_sets import FEATURE_ORDER, features_to_vector, pair_set_features
from .triads import TRIAD_FEATURE_ORDER, triad_features


def _as_set(X, name):
    """Accept a single embedding or a stack of them."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"{name} must be (d,) or (n, d); got {X.shape}")
    return X


def _save(obj, path):
    import pickle

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path):
    import pickle

    if not os.path.exists(path):
        raise FileNotFoundError(f"model not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_set_model(galleries, pairs, out_path, seed=42, max_fpr=0.25,
                    preprocessing="uncropped"):
    """Fit the set-level model.

    galleries: {identity: (n, d) array}
    pairs:     [(identity_a, identity_b, label)]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .calibration import calibrate_threshold

    X, y = [], []
    for a, b, label in pairs:
        A, B = galleries.get(a), galleries.get(b)
        if A is None or B is None:
            continue
        X.append(features_to_vector(pair_set_features(A, B)))
        y.append(label)
    X, y = np.stack(X), np.array(y)

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, random_state=seed)
    clf.fit(scaler.transform(X), y)

    # Calibrated on the fitting data here; the evaluation scripts calibrate on
    # a group-disjoint validation slice, which is what the reported numbers use.
    s = clf.predict_proba(scaler.transform(X))[:, 1]
    thr = calibrate_threshold(y, s, objective="accuracy", max_fpr=max_fpr)

    _save({"scaler": scaler, "clf": clf, "threshold": float(thr),
           "feature_order": list(FEATURE_ORDER), "kind": "set",
           "preprocessing": preprocessing}, out_path)
    return out_path


def train_triad_model(rows, out_path, seed=42, max_fpr=0.25,
                      preprocessing="uncropped"):
    """Fit the triadic model. rows: [(F, M, C, label)] as embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .calibration import calibrate_threshold

    X, y = [], []
    for F, M, C, label in rows:
        f = triad_features(F, M, C)
        X.append([f[k] for k in TRIAD_FEATURE_ORDER])
        y.append(label)
    X, y = np.array(X, dtype=np.float64), np.array(y)

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    s = clf.predict_proba(scaler.transform(X))[:, 1]
    thr = calibrate_threshold(y, s, objective="accuracy", max_fpr=max_fpr)

    _save({"scaler": scaler, "clf": clf, "threshold": float(thr),
           "feature_order": list(TRIAD_FEATURE_ORDER), "kind": "triad",
           "preprocessing": preprocessing}, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class SetKinshipPredictor:
    """Score two people, each represented by one or more photographs."""

    def __init__(self, model_path):
        m = _load(model_path)
        if m.get("kind") != "set":
            raise ValueError(f"{model_path} is not a set-level model")
        self.scaler = m["scaler"]
        self.clf = m["clf"]
        self.threshold = float(m["threshold"])
        self.metrics = m.get("metrics", {})
        # How the fitting embeddings were produced. Serving embeddings made a
        # different way flipped 30.2% of verdicts on 387 person-pairs, so the
        # caller must match this rather than choose independently.
        self.preprocessing = m.get("preprocessing", "uncropped")

    def predict_sets(self, A, B):
        A, B = _as_set(A, "A"), _as_set(B, "B")
        if A.shape[1] != B.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: {A.shape[1]} vs {B.shape[1]}")

        # Kinship is symmetric; average both orders so the answer cannot depend
        # on which person the caller passed first.
        v1 = features_to_vector(pair_set_features(A, B))
        v2 = features_to_vector(pair_set_features(B, A))
        p = float(np.mean(
            self.clf.predict_proba(self.scaler.transform(np.stack([v1, v2])))[:, 1]))

        return {"probability": p,
                "is_kin": bool(p >= self.threshold),
                "threshold": self.threshold,
                "n_a": int(A.shape[0]),
                "n_b": int(B.shape[0])}

    def predict_images(self, paths_a, paths_b, extractor=None):
        """Score two people from lists of image paths."""
        if extractor is None:
            from .models_improved import FaceFeatureExtractor

            extractor = FaceFeatureExtractor()
        if isinstance(paths_a, str):
            paths_a = [paths_a]
        if isinstance(paths_b, str):
            paths_b = [paths_b]
        A = np.stack([extractor.extract(p) for p in paths_a])
        B = np.stack([extractor.extract(p) for p in paths_b])
        return self.predict_sets(A, B)


class TriadPredictor:
    """Score a father-mother-child triad jointly."""

    def __init__(self, model_path):
        m = _load(model_path)
        if m.get("kind") != "triad":
            raise ValueError(f"{model_path} is not a triadic model")
        self.scaler = m["scaler"]
        self.clf = m["clf"]
        self.threshold = float(m["threshold"])
        self.metrics = m.get("metrics", {})
        self.preprocessing = m.get("preprocessing", "uncropped")

    def predict_triad(self, F, M, C):
        f = triad_features(np.asarray(F).ravel(),
                           np.asarray(M).ravel(),
                           np.asarray(C).ravel())
        v = np.array([[f[k] for k in TRIAD_FEATURE_ORDER]], dtype=np.float64)
        p = float(self.clf.predict_proba(self.scaler.transform(v))[0, 1])
        return {"probability": p,
                "is_kin": bool(p >= self.threshold),
                "threshold": self.threshold,
                "alpha": f["alpha"],
                "resembles": "father" if f["alpha"] > 0.5 else "mother",
                "cos_fc": f["cos_fc"],
                "cos_mc": f["cos_mc"]}

    def predict_images(self, father, mother, child, extractor=None):
        if extractor is None:
            from .models_improved import FaceFeatureExtractor

            extractor = FaceFeatureExtractor()
        return self.predict_triad(extractor.extract(father),
                                  extractor.extract(mother),
                                  extractor.extract(child))
