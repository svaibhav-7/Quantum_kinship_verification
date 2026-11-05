import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.evaluation.rerun_formulations_tskinface import (
    _weighted_auc,
    _weighted_auc_plan,
    cluster_bootstrap_delta,
)


class TestFamilyClusterBootstrap(unittest.TestCase):
    def setUp(self):
        self.treatment = [{
            "fold": 1,
            "test_labels": [1, 0, 1, 0],
            "test_scores": [0.9, 0.2, 0.8, 0.4],
            "test_family_a": ["a", "b", "c", "d"],
            "test_family_b": ["a", "b", "c", "d"],
        }]
        self.control = [{
            **self.treatment[0],
            "test_scores": [0.8, 0.3, 0.5, 0.5],
        }]

    def test_bootstrap_is_reproducible_and_returns_family_interval(self):
        first = cluster_bootstrap_delta(
            self.treatment, self.control, seed=7, replicates=500
        )
        second = cluster_bootstrap_delta(
            self.treatment, self.control, seed=7, replicates=500
        )

        self.assertEqual(first, second)
        self.assertEqual(first["cluster_unit"], "family")
        self.assertGreaterEqual(first["valid_replicates"], 250)
        self.assertLessEqual(first["valid_replicates"], 500)
        self.assertEqual(len(first["ci95"]), 2)

    def test_bootstrap_rejects_mismatched_test_groups(self):
        mismatched = [{
            **self.control[0],
            "test_family_a": ["a", "x", "c", "d"],
        }]
        with self.assertRaisesRegex(ValueError, "identical test pairs"):
            cluster_bootstrap_delta(
                self.treatment, mismatched, seed=7, replicates=100
            )

    def test_weighted_auc_matches_sklearn_with_ties(self):
        labels = np.asarray([1, 0, 1, 0, 1, 0])
        scores = np.asarray([0.8, 0.8, 0.4, 0.4, 0.4, 0.1])
        weights = np.asarray([2, 1, 3, 0, 1, 2], dtype=np.float64)

        actual = _weighted_auc(_weighted_auc_plan(labels, scores), weights)
        expected = roc_auc_score(labels, scores, sample_weight=weights)

        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
