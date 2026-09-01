"""Triadic evaluation must build labelled triads without leaking families
across folds, and must fall back cleanly when a triad is unavailable."""
import os
import sys
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts", "evaluation"))

from src.ts_pairs import family_of_ts


class TestTriadSampleConstruction(unittest.TestCase):
    def setUp(self):
        from run_triadic import build_triad_samples
        self.build = build_triad_samples
        self.triads = {
            f"FMS-{i}": {"F": f"x/FMS-{i}-F.jpg",
                         "M": f"x/FMS-{i}-M.jpg",
                         "C": f"x/FMS-{i}-S.jpg"}
            for i in range(1, 21)
        }

    def test_produces_balanced_positives_and_negatives(self):
        s = self.build(self.triads, seed=0)
        pos = [x for x in s if x[3] == 1]
        neg = [x for x in s if x[3] == 0]
        self.assertEqual(len(pos), len(neg))
        self.assertEqual(len(pos), len(self.triads))

    def test_positive_keeps_the_true_child(self):
        s = self.build(self.triads, seed=0)
        for F, M, C, label, fam in s:
            if label == 1:
                self.assertEqual(family_of_ts(C), fam)

    def test_negative_swaps_in_a_child_from_another_family(self):
        s = self.build(self.triads, seed=0)
        for F, M, C, label, fam in s:
            if label == 0:
                self.assertNotEqual(family_of_ts(C), fam,
                                    "negative reused the true child")
                self.assertEqual(family_of_ts(F), fam,
                                 "parents must stay with their own family")

    def test_every_sample_is_attributed_to_the_parents_family(self):
        """Fold assignment keys on this; a wrong key leaks across folds."""
        for F, M, C, label, fam in self.build(self.triads, seed=0):
            self.assertEqual(family_of_ts(F), fam)
            self.assertEqual(family_of_ts(M), fam)

    def test_is_deterministic_for_a_seed(self):
        a = self.build(self.triads, seed=3)
        b = self.build(self.triads, seed=3)
        self.assertEqual(a, b)

    def test_single_family_yields_no_negatives(self):
        one = {"FMS-1": self.triads["FMS-1"]}
        s = self.build(one, seed=0)
        self.assertTrue(all(x[3] == 1 for x in s))


if __name__ == "__main__":
    unittest.main()
