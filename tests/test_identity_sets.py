"""Identity photo sets.

FIW carries a median of 5 photos per identity; the pipeline used one and
discarded the rest. Probes measured that mean-pooling the set lifts raw
separability from 0.734 to 0.805 ROC-AUC with no training.

The hard requirement is graceful degradation: a set of size 1 must reduce
EXACTLY to the single-image path, so KinFaceW (one photo per person) is
unaffected. Any drift there is a bug, not a result.
"""
import os
import sys
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.identity_sets import (build_identity_sets, identity_of_path,
                               pair_set_features, set_descriptor)


def _p(fid, mid, img):
    return os.path.join("public", "FIDs", fid, f"MID{mid}", f"{img}.jpg")


class TestBuildIdentitySets(unittest.TestCase):
    def test_groups_images_by_identity(self):
        pairs = [
            (_p("F1", 1, "a"), _p("F1", 2, "x"), 1, "fd"),
            (_p("F1", 1, "b"), _p("F1", 2, "y"), 1, "fd"),
        ]
        sets = build_identity_sets(pairs)
        self.assertEqual(len(sets["F1/MID1"]), 2)
        self.assertEqual(len(sets["F1/MID2"]), 2)

    def test_deduplicates_repeated_images(self):
        same = _p("F1", 1, "a")
        pairs = [(same, _p("F1", 2, "x"), 1, "fd")] * 3
        sets = build_identity_sets(pairs)
        self.assertEqual(len(sets["F1/MID1"]), 1)

    def test_separates_identities_across_families(self):
        pairs = [
            (_p("F1", 1, "a"), _p("F1", 2, "x"), 1, "fd"),
            (_p("F2", 1, "a"), _p("F2", 2, "x"), 1, "fd"),
        ]
        sets = build_identity_sets(pairs)
        self.assertIn("F1/MID1", sets)
        self.assertIn("F2/MID1", sets)
        self.assertEqual(len(sets), 4)


class TestSetDescriptor(unittest.TestCase):
    def test_mean_is_l2_normalised(self):
        X = np.random.default_rng(0).normal(size=(5, 8))
        d = set_descriptor(X)
        self.assertAlmostEqual(float(np.linalg.norm(d["mean"])), 1.0, places=5)

    def test_singleton_set_has_zero_spread(self):
        X = np.random.default_rng(1).normal(size=(1, 8))
        d = set_descriptor(X)
        self.assertEqual(d["n"], 1)
        self.assertAlmostEqual(d["spread"], 0.0, places=6)

    def test_singleton_mean_is_the_normalised_image(self):
        """Graceful degradation: one photo in, that photo out."""
        X = np.random.default_rng(2).normal(size=(1, 16))
        expect = X[0] / np.linalg.norm(X[0])
        np.testing.assert_allclose(set_descriptor(X)["mean"], expect, atol=1e-6)

    def test_identical_images_give_zero_spread(self):
        v = np.random.default_rng(3).normal(size=16)
        X = np.stack([v, v, v])
        self.assertAlmostEqual(set_descriptor(X)["spread"], 0.0, places=5)

    def test_varied_images_give_positive_spread(self):
        X = np.random.default_rng(4).normal(size=(6, 16))
        self.assertGreater(set_descriptor(X)["spread"], 0.0)


class TestPairSetFeatures(unittest.TestCase):
    def test_singleton_pair_reduces_to_plain_cosine(self):
        """THE degradation guarantee. Every cross-set statistic must collapse
        to the single cosine when both sets have one image."""
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(1, 32)), rng.normal(size=(1, 32))
        an = a[0] / np.linalg.norm(a[0])
        bn = b[0] / np.linalg.norm(b[0])
        cos = float(an @ bn)

        f = pair_set_features(a, b)
        for key in ("cos_mean", "x_max", "x_min", "x_mean"):
            self.assertAlmostEqual(f[key], cos, places=5,
                                   msg=f"{key} did not reduce to the cosine")
        self.assertAlmostEqual(f["x_std"], 0.0, places=6)
        self.assertAlmostEqual(f["spread_a"], 0.0, places=6)
        self.assertAlmostEqual(f["spread_b"], 0.0, places=6)

    def test_feature_vector_is_symmetric_in_set_order(self):
        rng = np.random.default_rng(6)
        A, B = rng.normal(size=(4, 32)), rng.normal(size=(3, 32))
        f1 = pair_set_features(A, B)
        f2 = pair_set_features(B, A)
        for k in ("cos_mean", "x_max", "x_min", "x_mean", "x_std"):
            self.assertAlmostEqual(f1[k], f2[k], places=6, msg=k)

    def test_returns_finite_values_for_all_set_sizes(self):
        rng = np.random.default_rng(7)
        for na in (1, 2, 7):
            for nb in (1, 3, 5):
                f = pair_set_features(rng.normal(size=(na, 32)),
                                      rng.normal(size=(nb, 32)))
                for k, v in f.items():
                    self.assertTrue(np.isfinite(v), f"{k} not finite at {na}x{nb}")

    def test_matching_sets_score_higher_than_random_sets(self):
        """Sanity: the features must carry signal, not just be well-formed."""
        rng = np.random.default_rng(8)
        base = rng.normal(size=32)
        A = base + 0.1 * rng.normal(size=(5, 32))
        B = base + 0.1 * rng.normal(size=(5, 32))
        C = rng.normal(size=(5, 32))
        self.assertGreater(pair_set_features(A, B)["cos_mean"],
                           pair_set_features(A, C)["cos_mean"])

    def test_density_arm_is_off_by_default(self):
        rng = np.random.default_rng(9)
        f = pair_set_features(rng.normal(size=(4, 32)), rng.normal(size=(4, 32)))
        self.assertNotIn("density_fid", f)

    def test_density_arm_available_when_requested(self):
        """Pre-registered control: available, measurable, off by default."""
        rng = np.random.default_rng(10)
        f = pair_set_features(rng.normal(size=(4, 32)), rng.normal(size=(4, 32)),
                              with_density=True)
        self.assertIn("density_fid", f)
        self.assertTrue(0.0 <= f["density_fid"] <= 1.0)




class TestIdentityKeyIsDatasetAware(unittest.TestCase):
    """identity_of_path was written for FIW's FIDs/<fam>/MID<n>/ layout. On
    KinFaceW it returned the directory ('images/father-dau'), collapsing all
    533 people into 4 buckets and manufacturing set sizes of 55-169 for a
    corpus with exactly ONE photo per person. That produced a spurious
    0.725 -> 0.898 AUC 'gain'."""

    def test_kinfacew_each_person_is_its_own_identity(self):
        a = os.path.join("KinFaceW-I", "images", "father-dau", "fd_024_1.jpg")
        b = os.path.join("KinFaceW-I", "images", "father-dau", "fd_024_2.jpg")
        c = os.path.join("KinFaceW-I", "images", "father-dau", "fd_025_1.jpg")
        ia, ib, ic = (identity_of_path(x) for x in (a, b, c))
        self.assertNotEqual(ia, ib, "parent and child merged into one identity")
        self.assertNotEqual(ia, ic, "different families merged")
        self.assertEqual(len({ia, ib, ic}), 3)

    def test_kinfacew_sets_are_singletons(self):
        pairs = [
            (os.path.join("KinFaceW-I", "images", "father-dau", f"fd_{i:03d}_1.jpg"),
             os.path.join("KinFaceW-I", "images", "father-dau", f"fd_{i:03d}_2.jpg"),
             1, "fd")
            for i in range(1, 21)
        ]
        sets = build_identity_sets(pairs)
        self.assertEqual(len(sets), 40, "expected one identity per image")
        for ident, paths in sets.items():
            self.assertEqual(len(paths), 1, f"{ident} should hold one photo")

    def test_tskinface_each_role_is_its_own_identity(self):
        f = os.path.join("TSKinFace_Data", "x", "FMS-12-F.jpg")
        m = os.path.join("TSKinFace_Data", "x", "FMS-12-M.jpg")
        self.assertNotEqual(identity_of_path(f), identity_of_path(m))

    def test_fiw_still_groups_photos_of_one_person(self):
        """The FIW behaviour must be preserved -- that is where the gain is."""
        a = _p("F0282", 3, "P01")
        b = _p("F0282", 3, "P02")
        c = _p("F0282", 4, "P01")
        self.assertEqual(identity_of_path(a), identity_of_path(b))
        self.assertNotEqual(identity_of_path(a), identity_of_path(c))


if __name__ == "__main__":
    unittest.main()


class TestSpreadIsComputedInClosedForm(unittest.TestCase):
    """The mean off-diagonal Gram entry has a closed form:

        sum_{i!=j} <xi,xj> = ||sum_i xi||^2 - n     (unit vectors)

    The original implementation materialised triu_indices, which allocated
    ~n^2/2 index pairs and dominated the cost at large set sizes (46 ms of a
    48 ms call at n=160). These tests pin the closed form against the explicit
    computation so the optimisation cannot silently change the value.
    """

    def _explicit(self, X):
        # Must use the same normalisation path as set_descriptor: comparing
        # against a float32 reference showed a spurious 1.4e-07 mismatch that
        # was an artefact of the check, not of the optimisation.
        from src.identity_sets import _unit

        Xn = _unit(X)
        n = Xn.shape[0]
        G = Xn @ Xn.T
        iu = np.triu_indices(n, k=1)
        return float(1.0 - G[iu].mean())

    def test_matches_explicit_triu_computation(self):
        rng = np.random.default_rng(0)
        for n in (2, 3, 5, 17, 64, 233):
            X = rng.normal(size=(n, 32))
            self.assertAlmostEqual(set_descriptor(X)["spread"],
                                   max(0.0, self._explicit(X)), places=11,
                                   msg=f"n={n}")

    def test_matches_on_real_embeddings(self):
        """Synthetic Gaussians are an easy case; real embeddings are clustered
        and expose cancellation that random vectors do not."""
        import pickle

        cache_path = os.path.join(project_root, "weights", "caches",
                                  "all_datasets_cache.pkl")
        if not os.path.exists(cache_path):
            raise unittest.SkipTest("embedding cache not available")
        vals = list(pickle.load(open(cache_path, "rb")).values())
        rng = np.random.default_rng(0)
        for n in (2, 5, 21, 89, 233):
            idx = rng.choice(len(vals), n, replace=False)
            X = np.stack([np.asarray(vals[i]) for i in idx])
            self.assertAlmostEqual(set_descriptor(X)["spread"],
                                   max(0.0, self._explicit(X)), places=10,
                                   msg=f"n={n}")

    def test_matches_for_near_identical_vectors(self):
        """Numerically the hardest case: the closed form subtracts two nearly
        equal quantities, so cancellation would show up here first."""
        rng = np.random.default_rng(1)
        base = rng.normal(size=64)
        X = base + 1e-6 * rng.normal(size=(20, 64))
        self.assertAlmostEqual(set_descriptor(X)["spread"],
                               max(0.0, self._explicit(X)), places=8)

    def test_still_zero_for_singleton_and_identical_sets(self):
        rng = np.random.default_rng(2)
        self.assertEqual(set_descriptor(rng.normal(size=(1, 16)))["spread"], 0.0)
        v = rng.normal(size=16)
        self.assertAlmostEqual(set_descriptor(np.stack([v] * 4))["spread"],
                               0.0, places=9)
