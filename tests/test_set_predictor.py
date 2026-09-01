"""Deployment path for the set-level and triadic models.

Set-level was measured at +0.077 ROC-AUC on FIW and exactly 0.000 on the three
single-photograph corpora. The deployed predictor must reproduce both halves of
that: a real gain when several photos are supplied, and bit-identical output to
the single-image path when only one is.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.set_predictor import SetKinshipPredictor, TriadPredictor


def _fit_toy_set_model(path, seed=0):
    """Train a tiny set-level model on synthetic data and save it."""
    from src.set_predictor import train_set_model

    rng = np.random.default_rng(seed)
    galleries, pairs = {}, []
    for i in range(60):
        base = rng.normal(size=64)
        galleries[f"p{i}a"] = base + 0.15 * rng.normal(size=(4, 64))
        galleries[f"p{i}b"] = base + 0.15 * rng.normal(size=(3, 64))
        galleries[f"q{i}"] = rng.normal(size=(4, 64))
        pairs.append((f"p{i}a", f"p{i}b", 1))
        pairs.append((f"p{i}a", f"q{i}", 0))
    train_set_model(galleries, pairs, path, seed=seed)
    return path


class TestSetPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls._tmp.name, "set_model.joblib")
        _fit_toy_set_model(cls.path)

    def test_loads_and_reports_threshold(self):
        p = SetKinshipPredictor(self.path)
        self.assertTrue(0.0 <= p.threshold <= 1.0)

    def test_scores_a_set_pair(self):
        p = SetKinshipPredictor(self.path)
        rng = np.random.default_rng(1)
        r = p.predict_sets(rng.normal(size=(5, 64)), rng.normal(size=(3, 64)))
        self.assertIn("probability", r)
        self.assertIn("is_kin", r)
        self.assertTrue(0.0 <= r["probability"] <= 1.0)
        self.assertEqual(r["n_a"], 5)
        self.assertEqual(r["n_b"], 3)

    def test_single_image_input_is_accepted(self):
        """Graceful degradation at the API level: a 1-D array is one photo."""
        p = SetKinshipPredictor(self.path)
        rng = np.random.default_rng(2)
        r = p.predict_sets(rng.normal(size=64), rng.normal(size=64))
        self.assertEqual(r["n_a"], 1)
        self.assertEqual(r["n_b"], 1)
        self.assertTrue(0.0 <= r["probability"] <= 1.0)

    def test_is_symmetric_in_argument_order(self):
        p = SetKinshipPredictor(self.path)
        rng = np.random.default_rng(3)
        A, B = rng.normal(size=(4, 64)), rng.normal(size=(2, 64))
        self.assertAlmostEqual(p.predict_sets(A, B)["probability"],
                               p.predict_sets(B, A)["probability"], places=6)

    def test_more_photos_of_the_same_person_do_not_break_scoring(self):
        p = SetKinshipPredictor(self.path)
        rng = np.random.default_rng(4)
        base = rng.normal(size=64)
        A = base + 0.1 * rng.normal(size=(6, 64))
        B = base + 0.1 * rng.normal(size=(6, 64))
        C = rng.normal(size=(6, 64))
        self.assertGreater(p.predict_sets(A, B)["probability"],
                           p.predict_sets(A, C)["probability"])

    def test_decision_uses_the_calibrated_threshold(self):
        p = SetKinshipPredictor(self.path)
        p.threshold = 0.99
        rng = np.random.default_rng(5)
        r = p.predict_sets(rng.normal(size=(3, 64)), rng.normal(size=(3, 64)))
        self.assertEqual(r["is_kin"], r["probability"] >= 0.99)

    def test_rejects_mismatched_embedding_dimension(self):
        p = SetKinshipPredictor(self.path)
        rng = np.random.default_rng(6)
        with self.assertRaises(ValueError):
            p.predict_sets(rng.normal(size=(3, 64)), rng.normal(size=(3, 32)))


class TestTriadPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.set_predictor import train_triad_model

        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls._tmp.name, "triad_model.joblib")
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(80):
            F, M = rng.normal(size=64), rng.normal(size=64)
            C = 0.5 * F + 0.5 * M + 0.15 * rng.normal(size=64)
            rows.append((F, M, C, 1))
            rows.append((F, M, rng.normal(size=64), 0))
        train_triad_model(rows, cls.path, seed=0)

    def test_scores_a_triad(self):
        p = TriadPredictor(self.path)
        rng = np.random.default_rng(7)
        F, M = rng.normal(size=64), rng.normal(size=64)
        C = 0.5 * F + 0.5 * M + 0.1 * rng.normal(size=64)
        r = p.predict_triad(F, M, C)
        self.assertTrue(0.0 <= r["probability"] <= 1.0)
        self.assertIn("alpha", r)

    def test_true_child_scores_above_unrelated_child(self):
        p = TriadPredictor(self.path)
        rng = np.random.default_rng(8)
        F, M = rng.normal(size=64), rng.normal(size=64)
        C = 0.5 * F + 0.5 * M + 0.1 * rng.normal(size=64)
        U = rng.normal(size=64)
        self.assertGreater(p.predict_triad(F, M, C)["probability"],
                           p.predict_triad(F, M, U)["probability"])

    def test_reports_which_parent_the_child_resembles(self):
        p = TriadPredictor(self.path)
        rng = np.random.default_rng(9)
        F, M = rng.normal(size=64), rng.normal(size=64)
        father_like = F + 0.05 * rng.normal(size=64)
        self.assertGreater(p.predict_triad(F, M, father_like)["alpha"], 0.5)


if __name__ == "__main__":
    unittest.main()
