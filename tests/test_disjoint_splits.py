"""Splits must be family-disjoint: no family, and therefore no identity,
may appear on both sides. This is the property the old random pair-level
split violated on 100% of test pairs."""
import os
import sys
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.splits import family_of, identity_of, family_disjoint_split


def _pair(fid, mid1, mid2, label=1):
    a = os.path.join("public", "FIDs", fid, f"MID{mid1}", "img.jpg")
    b = os.path.join("public", "FIDs", fid, f"MID{mid2}", "img.jpg")
    return (a, b, label, "fd")


class TestPathParsing(unittest.TestCase):
    def test_family_of_extracts_family_id(self):
        p = os.path.join("public", "FIDs", "F0282", "MID3", "P01.jpg")
        self.assertEqual(family_of(p), "F0282")

    def test_identity_of_is_unique_per_person_within_family(self):
        a = os.path.join("public", "FIDs", "F0282", "MID3", "P01.jpg")
        b = os.path.join("public", "FIDs", "F0282", "MID3", "P02.jpg")
        c = os.path.join("public", "FIDs", "F0282", "MID4", "P01.jpg")
        self.assertEqual(identity_of(a), identity_of(b))
        self.assertNotEqual(identity_of(a), identity_of(c))


class TestFamilyDisjointSplit(unittest.TestCase):
    def setUp(self):
        self.pairs = []
        for f in range(20):
            for k in range(5):
                self.pairs.append(_pair(f"F{f:04d}", 1, 2 + k))

    def test_no_family_appears_in_both_splits(self):
        train, test = family_disjoint_split(self.pairs, test_ratio=0.3, seed=0)
        tr_f = {family_of(p[0]) for p in train} | {family_of(p[1]) for p in train}
        te_f = {family_of(p[0]) for p in test} | {family_of(p[1]) for p in test}
        self.assertTrue(tr_f.isdisjoint(te_f))

    def test_no_identity_appears_in_both_splits(self):
        train, test = family_disjoint_split(self.pairs, test_ratio=0.3, seed=0)
        tr_i = {identity_of(p[0]) for p in train} | {identity_of(p[1]) for p in train}
        te_i = {identity_of(p[0]) for p in test} | {identity_of(p[1]) for p in test}
        self.assertTrue(tr_i.isdisjoint(te_i))

    def test_every_pair_is_kept_exactly_once(self):
        train, test = family_disjoint_split(self.pairs, test_ratio=0.3, seed=0)
        self.assertEqual(len(train) + len(test), len(self.pairs))

    def test_both_splits_are_non_empty(self):
        train, test = family_disjoint_split(self.pairs, test_ratio=0.3, seed=0)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_split_is_deterministic_for_a_seed(self):
        a1, b1 = family_disjoint_split(self.pairs, test_ratio=0.3, seed=7)
        a2, b2 = family_disjoint_split(self.pairs, test_ratio=0.3, seed=7)
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)

    def test_cross_family_pair_is_never_split_across_sides(self):
        """Non-kin pairs join two families; that pair must not leak a family."""
        pairs = list(self.pairs)
        a = os.path.join("public", "FIDs", "F0000", "MID1", "x.jpg")
        b = os.path.join("public", "FIDs", "F0019", "MID1", "y.jpg")
        pairs.append((a, b, 0, "fd"))
        train, test = family_disjoint_split(pairs, test_ratio=0.3, seed=0)
        tr_f = {family_of(p[0]) for p in train} | {family_of(p[1]) for p in train}
        te_f = {family_of(p[0]) for p in test} | {family_of(p[1]) for p in test}
        self.assertTrue(tr_f.isdisjoint(te_f))


if __name__ == "__main__":
    unittest.main()


class TestSplitThenBuild(unittest.TestCase):
    """Non-kin pairs join two families, which transitively links the whole
    dataset into one component (measured on real FIW: 1 component, 100% of
    pairs). So negatives must be generated per-side, after the family split.
    """

    def setUp(self):
        # Kin pairs only; negatives are built by the function under test.
        self.kin = []
        for f in range(20):
            for k in range(4):
                self.kin.append(_pair(f"F{f:04d}", 1, 2 + k, label=1))

    def test_splits_families_and_builds_balanced_negatives_per_side(self):
        from src.splits import split_then_build

        train, test = split_then_build(self.kin, test_ratio=0.3, seed=0)

        for name, side in (("train", train), ("test", test)):
            kin = [p for p in side if p[2] == 1]
            non = [p for p in side if p[2] == 0]
            self.assertGreater(len(kin), 0, f"{name} has no positives")
            self.assertGreater(len(non), 0, f"{name} has no negatives")
            self.assertEqual(len(kin), len(non), f"{name} is not 1:1 balanced")

    def test_negatives_never_cross_the_split_boundary(self):
        from src.splits import split_then_build

        train, test = split_then_build(self.kin, test_ratio=0.3, seed=0)
        tr_f = {family_of(p[0]) for p in train} | {family_of(p[1]) for p in train}
        te_f = {family_of(p[0]) for p in test} | {family_of(p[1]) for p in test}
        self.assertTrue(
            tr_f.isdisjoint(te_f),
            "a generated negative bridged the two sides",
        )

    def test_negatives_are_cross_family_so_label_is_correct(self):
        from src.splits import split_then_build

        train, test = split_then_build(self.kin, test_ratio=0.3, seed=0)
        for side in (train, test):
            for a, b, label, _ in side:
                if label == 0:
                    self.assertNotEqual(
                        family_of(a), family_of(b),
                        "a 'non-kin' pair came from one family",
                    )
