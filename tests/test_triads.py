"""Triadic kinship features.

TSKinFace supplies father + mother + child, and the pairwise pipeline splits
each triad into two independent pairs. Probes measured that scoring the triad
jointly lifts separability from 0.761 to 0.792 ROC-AUC -- and that most of the
gain comes from parental similarity, a quantity no pairwise model can see.
"""
import os
import sys
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.triads import find_triads, triad_features, TRIAD_FEATURE_ORDER


class TestFindTriads(unittest.TestCase):
    def setUp(self):
        self.root = os.path.join(project_root, "TSKinFace_Data",
                                 "TSKinFace_Data", "TSKinFace_cropped")
        if not os.path.exists(self.root):
            raise unittest.SkipTest("TSKinFace not available")

    def test_finds_complete_triads_only(self):
        triads = find_triads(self.root)
        self.assertGreater(len(triads), 100)
        for fam, members in triads.items():
            self.assertEqual(set(members), {"F", "M", "C"})
            for path in members.values():
                self.assertTrue(os.path.exists(path))

    def test_all_three_members_share_a_family(self):
        from src.ts_pairs import family_of_ts
        for fam, m in find_triads(self.root).items():
            fams = {family_of_ts(p) for p in m.values()}
            self.assertEqual(len(fams), 1)
            self.assertEqual(fams.pop(), fam)


class TestTriadFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.F = rng.normal(size=64)
        self.M = rng.normal(size=64)
        # a child resembling both parents
        self.C = 0.5 * self.F + 0.5 * self.M + 0.1 * rng.normal(size=64)
        self.U = rng.normal(size=64)  # unrelated

    def test_returns_every_declared_feature(self):
        f = triad_features(self.F, self.M, self.C)
        for k in TRIAD_FEATURE_ORDER:
            self.assertIn(k, f)
            self.assertTrue(np.isfinite(f[k]), f"{k} not finite")

    def test_mixture_explains_a_real_child_better_than_either_parent(self):
        """The premise of triadic scoring: a child resembles the PAIR."""
        f = triad_features(self.F, self.M, self.C)
        self.assertGreaterEqual(f["best_mixture"], max(f["cos_fc"], f["cos_mc"]))

    def test_scores_a_true_child_above_an_unrelated_child(self):
        real = triad_features(self.F, self.M, self.C)
        fake = triad_features(self.F, self.M, self.U)
        self.assertGreater(real["best_mixture"], fake["best_mixture"])

    def test_alpha_reports_which_parent_dominates(self):
        rng = np.random.default_rng(1)
        F = rng.normal(size=64)
        M = rng.normal(size=64)
        father_like = F + 0.05 * rng.normal(size=64)
        f = triad_features(F, M, father_like)
        self.assertGreater(f["alpha"], 0.5)

    def test_includes_parental_similarity(self):
        """cos(F,M) alone contributed +2.2 AUC in probe 3; a pairwise model
        cannot express it."""
        f = triad_features(self.F, self.M, self.C)
        n = lambda v: v / np.linalg.norm(v)
        self.assertAlmostEqual(f["cos_fm"], float(n(self.F) @ n(self.M)), places=5)

    def test_is_invariant_to_embedding_scale(self):
        f1 = triad_features(self.F, self.M, self.C)
        f2 = triad_features(3.0 * self.F, 0.2 * self.M, 7.0 * self.C)
        for k in ("cos_fc", "cos_mc", "cos_fm", "best_mixture"):
            self.assertAlmostEqual(f1[k], f2[k], places=5, msg=k)


if __name__ == "__main__":
    unittest.main()
