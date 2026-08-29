"""TSKinFace's stock negatives are 100% cross-family while its positives are
100% same-family, so 'same photo session' alone separates the classes. Honest
negatives must include same-family non-kin pairs (father-vs-mother), which
share the session but not the kinship."""
import os, sys, unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ts_pairs import build_tskinface_pairs, family_of_ts, role_of_ts


class TestTSPathHelpers(unittest.TestCase):
    def test_extracts_family_and_role(self):
        p = os.path.join("x", "FMS-12-F.jpg")
        self.assertEqual(family_of_ts(p), "FMS-12")
        self.assertEqual(role_of_ts(p), "F")


class TestHonestNegatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.join(project_root, "TSKinFace_Data", "TSKinFace_Data",
                            "TSKinFace_cropped")
        if not os.path.exists(root):
            raise unittest.SkipTest("TSKinFace not available")
        cls.pairs = build_tskinface_pairs(root, same_family_negative_ratio=0.5)

    def test_produces_balanced_pairs(self):
        kin = [p for p in self.pairs if p[2] == 1]
        non = [p for p in self.pairs if p[2] == 0]
        self.assertGreater(len(kin), 0)
        self.assertAlmostEqual(len(kin), len(non), delta=max(2, len(kin) * 0.02))

    def test_some_negatives_are_same_family(self):
        """This is the whole point: negatives that share the photo session."""
        non = [p for p in self.pairs if p[2] == 0]
        same = [p for p in non if family_of_ts(p[0]) == family_of_ts(p[1])]
        self.assertGreater(len(same) / len(non), 0.3,
                           "too few same-family negatives to close the shortcut")

    def test_same_family_negatives_are_genuinely_not_kin(self):
        """Father-vs-Mother share a family but are not parent-child."""
        for a, b, label, _ in self.pairs:
            if label == 0 and family_of_ts(a) == family_of_ts(b):
                self.assertEqual({role_of_ts(a), role_of_ts(b)}, {"F", "M"},
                                 "same-family negative must be father-vs-mother")

    def test_positives_remain_parent_child(self):
        for a, b, label, _ in self.pairs:
            if label == 1:
                roles = {role_of_ts(a), role_of_ts(b)}
                self.assertTrue(roles & {"F", "M"} and roles & {"S", "D"})
                self.assertEqual(family_of_ts(a), family_of_ts(b))

    def test_family_membership_no_longer_predicts_the_label(self):
        kin = [p for p in self.pairs if p[2] == 1]
        non = [p for p in self.pairs if p[2] == 0]
        f_kin = sum(1 for p in kin if family_of_ts(p[0]) == family_of_ts(p[1])) / len(kin)
        f_non = sum(1 for p in non if family_of_ts(p[0]) == family_of_ts(p[1])) / len(non)
        # Stock loader gives 1.0 vs 0.0 -- a perfect shortcut.
        self.assertLess(abs(f_kin - f_non), 0.75,
                        "same-family still separates the classes almost perfectly")


if __name__ == "__main__":
    unittest.main()
