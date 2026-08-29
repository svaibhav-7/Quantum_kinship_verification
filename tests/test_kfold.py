"""Grouped k-fold: every group must be tested exactly once, and folds must
stay group-disjoint. This is what replaces the single lucky split."""
import os, sys, unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.kfold import grouped_kfold, balance_group_sizes


def _pairs(n_groups=20, per=6):
    out = []
    for g in range(n_groups):
        for i in range(per):
            out.append((f"KinFaceW-I/images/fd/fd_{g:03d}_1.jpg",
                        f"KinFaceW-I/images/fd/fd_{g:03d}_2.jpg", 1, "fd"))
    return out


class TestGroupedKFold(unittest.TestCase):
    def setUp(self):
        self.pairs = _pairs()
        self.key = lambda p: os.path.basename(p).rsplit("_", 1)[0]

    def test_yields_requested_number_of_folds(self):
        folds = list(grouped_kfold(self.pairs, self.key, n_splits=5, seed=0))
        self.assertEqual(len(folds), 5)

    def test_each_fold_is_group_disjoint(self):
        for tr, te in grouped_kfold(self.pairs, self.key, n_splits=5, seed=0):
            gt = {self.key(p[0]) for p in tr} | {self.key(p[1]) for p in tr}
            ge = {self.key(p[0]) for p in te} | {self.key(p[1]) for p in te}
            self.assertTrue(gt.isdisjoint(ge))

    def test_every_group_is_tested_exactly_once(self):
        seen = []
        for _tr, te in grouped_kfold(self.pairs, self.key, n_splits=5, seed=0):
            seen += sorted({self.key(p[0]) for p in te})
        all_groups = {self.key(p[0]) for p in self.pairs}
        self.assertEqual(sorted(seen), sorted(all_groups))
        self.assertEqual(len(seen), len(set(seen)), "a group was tested twice")

    def test_folds_are_deterministic(self):
        a = [sorted({self.key(p[0]) for p in te})
             for _t, te in grouped_kfold(self.pairs, self.key, 5, seed=1)]
        b = [sorted({self.key(p[0]) for p in te})
             for _t, te in grouped_kfold(self.pairs, self.key, 5, seed=1)]
        self.assertEqual(a, b)

    def test_no_fold_is_empty(self):
        for tr, te in grouped_kfold(self.pairs, self.key, n_splits=5, seed=0):
            self.assertGreater(len(tr), 0); self.assertGreater(len(te), 0)


class TestBalanceGroupSizes(unittest.TestCase):
    """FIW's F0282 is 19% of all pairs and 95% of one test split. Capping
    per-group pairs stops one family dominating any fold."""

    def setUp(self):
        self.pairs = ([("public/FIDs/F0282/MID1/a.jpg",
                        "public/FIDs/F0282/MID2/b.jpg", 1, "fd")] * 500
                      + [("public/FIDs/F0010/MID1/a.jpg",
                          "public/FIDs/F0010/MID2/b.jpg", 1, "fd")] * 10)
        self.key = lambda p: p.split("/")[2]

    def test_caps_the_dominant_group(self):
        out = balance_group_sizes(self.pairs, self.key, max_per_group=50, seed=0)
        from collections import Counter
        c = Counter(self.key(p[0]) for p in out)
        self.assertLessEqual(c["F0282"], 50)

    def test_leaves_small_groups_untouched(self):
        out = balance_group_sizes(self.pairs, self.key, max_per_group=50, seed=0)
        from collections import Counter
        c = Counter(self.key(p[0]) for p in out)
        self.assertEqual(c["F0010"], 10)


if __name__ == "__main__":
    unittest.main()
