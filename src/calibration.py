"""Decision-threshold calibration.

Youden's J maximises TPR-FPR, weighting both errors equally. For kinship
verification that produced a 24.3% false-positive rate on held-out FIW -- one
in four unrelated pairs called kin -- which is the wrong trade for most
deployments. This module lets the operating point be chosen explicitly.

Always calibrate on a validation split that is family-disjoint from test.
"""

import numpy as np


def calibrate_threshold(y_true, scores, objective="accuracy", max_fpr=None,
                        smooth=False, smooth_tol=0.005):
    """Pick a decision threshold.

    objective: "accuracy" maximises accuracy; "youden" maximises TPR-FPR.
    max_fpr:   optional ceiling on false-positive rate. If no candidate meets
               it, the threshold with the lowest achievable FPR is returned so
               the caller always gets a usable value.
    smooth:    take the median of all thresholds within `smooth_tol` of the
               best score instead of the single argmax. The argmax overfits
               the validation sample -- across seeds it swung accuracy by up
               to 6.8 points while AUC moved only 0.038.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores).ravel()

    # Midpoints between observed scores, plus the extremes.
    uniq = np.unique(s)
    if uniq.size > 2000:  # keep calibration cheap on large validation sets
        uniq = np.quantile(uniq, np.linspace(0, 1, 2000))
    cands = np.unique(np.r_[uniq, (uniq[:-1] + uniq[1:]) / 2.0])

    pos, neg = (y == 1), (y == 0)
    n_pos, n_neg = max(1, pos.sum()), max(1, neg.sum())

    pred = s[None, :] >= cands[:, None]
    tpr = (pred & pos[None, :]).sum(1) / n_pos
    fpr = (pred & neg[None, :]).sum(1) / n_neg
    acc = ((pred == (y[None, :] == 1)).sum(1)) / y.size

    if max_fpr is not None:
        ok = fpr <= max_fpr
        if not ok.any():
            # Constraint unreachable: fall back to the strictest available.
            return float(cands[int(np.argmin(fpr))])
        cands, tpr, fpr, acc = cands[ok], tpr[ok], fpr[ok], acc[ok]

    score = acc if objective == "accuracy" else (tpr - fpr)
    if not smooth:
        return float(cands[int(np.argmax(score))])

    near = score >= (score.max() - smooth_tol)
    return float(np.median(cands[near]))


def operating_point(y_true, scores, threshold):
    """Report the full operating point for a threshold."""
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores).ravel()
    p = s >= threshold
    pos, neg = (y == 1), (y == 0)
    tp, fp = int((p & pos).sum()), int((p & neg).sum())
    fn, tn = int((~p & pos).sum()), int((~p & neg).sum())
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(1, y.size) * 100),
        "tpr_recall": float(tp / max(1, tp + fn) * 100),
        "fpr": float(fp / max(1, fp + tn) * 100),
        "precision": float(tp / max(1, tp + fp) * 100),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
