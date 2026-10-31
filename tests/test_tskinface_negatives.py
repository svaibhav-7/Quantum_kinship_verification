"""Regression tests for TSKinFace shortcut-free negative construction."""
import os
import sys
import tempfile
import unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ts_pairs import (
    add_tskinface_negatives,
    build_tskinface_pairs,
    family_of_ts,
    role_of_ts,
)


def _make_synthetic_root(root, n_families=6):
    for folder, child_role in (("FMS", "S"), ("FMD", "D")):
        folder_path = os.path.join(root, folder)
        os.makedirs(folder_path, exist_ok=True)
        for i in range(n_families):
            for role in ("F", "M", child_role):
                path = os.path.join(folder_path, f"{folder}-{i:02d}-{role}.jpg")
                with open(path, "wb") as handle:
                    handle.write(b"")


class TestTSPathHelpers(unittest.TestCase):
    def test_extracts_family_and_role(self):
        path = os.path.join("x", "FMS-12-F.jpg")
        self.assertEqual(family_of_ts(path), "FMS-12")
        self.assertEqual(role_of_ts(path), "F")


class TestPairBuilder(unittest.TestCase):
    def test_pair_builder_has_balanced_classes_and_same_family_negatives(self):
        with tempfile.TemporaryDirectory() as root:
            _make_synthetic_root(root)
            pairs = build_tskinface_pairs(root, 0.5, seed=7)
        positives = [p for p in pairs if p[2] == 1]
        negatives = [p for p in pairs if p[2] == 0]
        self.assertEqual(len(positives), len(negatives))
        same = [p for p in negatives if family_of_ts(p[0]) == family_of_ts(p[1])]
        self.assertGreater(len(same), 0)
        for a, b, _label, _rel in same:
            self.assertEqual({role_of_ts(a), role_of_ts(b)}, {"F", "M"})

    def test_invalid_negative_ratio_rejected_by_fold_sampler(self):
        with self.assertRaises(ValueError):
            add_tskinface_negatives([], same_family_negative_ratio=1.1)


class TestFoldNegativeBuilder(unittest.TestCase):
    def setUp(self):
        self.positives = []
        for i in range(4):
            base = f"FMS-{i:02d}"
            self.positives.extend([
                (f"{base}-F.jpg", f"{base}-S.jpg", 1, "fs"),
                (f"{base}-M.jpg", f"{base}-S.jpg", 1, "ms"),
            ])

    def test_retains_same_family_and_generates_cross_family_negatives(self):
        pairs = add_tskinface_negatives(
            self.positives, seed=11, same_family_negative_ratio=0.5)
        negatives = [p for p in pairs if p[2] == 0]
        same = [p for p in negatives
                if family_of_ts(p[0]) == family_of_ts(p[1])]
        cross = [p for p in negatives
                 if family_of_ts(p[0]) != family_of_ts(p[1])]
        self.assertEqual(len(negatives), len(self.positives))
        self.assertEqual(len(same), 4)
        self.assertEqual(len(cross), 4)
        allowed_families = {family_of_ts(p[0]) for p in self.positives}
        self.assertTrue(all(family_of_ts(a) in allowed_families
                            and family_of_ts(b) in allowed_families
                            for a, b, _label, _rel in negatives))
        for a, b, _label, _rel in same:
            self.assertEqual({role_of_ts(a), role_of_ts(b)}, {"F", "M"})

    def test_single_family_side_keeps_valid_internal_negative(self):
        pairs = add_tskinface_negatives(self.positives[:2], seed=2)
        negatives = [p for p in pairs if p[2] == 0]
        self.assertEqual(len(negatives), 1)
        self.assertTrue(all(family_of_ts(a) == family_of_ts(b)
                            for a, b, _label, _rel in negatives))

    def test_rejects_pairs_that_are_not_positive_only(self):
        with self.assertRaises(ValueError):
            add_tskinface_negatives(self.positives + [
                ("FMS-00-F.jpg", "FMS-01-S.jpg", 0, "fs")
            ])

    def test_rejects_positive_pair_crossing_family_boundary(self):
        invalid = list(self.positives)
        invalid[0] = ("FMS-00-F.jpg", "FMS-01-S.jpg", 1, "fs")
        with self.assertRaises(ValueError):
            add_tskinface_negatives(invalid)


class TestDatasetBackedPairBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                            "TSKinFace_cropped")
        if not os.path.exists(root):
            raise unittest.SkipTest("TSKinFace data not available")
        cls.pairs = build_tskinface_pairs(root, same_family_negative_ratio=0.5)

    def test_builds_nonempty_balanced_pairs(self):
        positives = [p for p in self.pairs if p[2] == 1]
        negatives = [p for p in self.pairs if p[2] == 0]
        self.assertGreater(len(positives), 0)
        self.assertEqual(len(positives), len(negatives))

    def test_same_family_negatives_are_father_mother_pairs(self):
        negatives = [p for p in self.pairs if p[2] == 0]
        same = [p for p in negatives
                if family_of_ts(p[0]) == family_of_ts(p[1])]
        self.assertGreater(len(same), 0)
        for a, b, _label, _rel in same:
            self.assertEqual({role_of_ts(a), role_of_ts(b)}, {"F", "M"})


if __name__ == "__main__":
    unittest.main()
