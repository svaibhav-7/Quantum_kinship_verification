"""Pooled multi-dataset splits must stay group-disjoint per dataset."""
import os, sys, unittest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.multi_dataset import load_all_datasets, _kfw_family, _split_by_key


class TestKfwFamily(unittest.TestCase):
    def test_parses_kinfacew_family(self):
        self.assertEqual(_kfw_family("x/fd_001_1.jpg"), "fd_001")
        self.assertEqual(_kfw_family("x/fd_001_2.jpg"), "fd_001")
        self.assertNotEqual(_kfw_family("x/fd_001_1.jpg"), _kfw_family("x/fd_002_1.jpg"))


class TestSplitByKey(unittest.TestCase):
    def test_groups_never_span_both_sides(self):
        pairs = [(f"g{i}_a.jpg", f"g{i}_b.jpg", i % 2, "fd") for i in range(40)]
        key = lambda p: os.path.basename(p).split("_")[0]
        tr, te = _split_by_key(pairs, key, 0.25, 0)
        self.assertTrue({key(p[0]) for p in tr}.isdisjoint({key(p[0]) for p in te}))
        self.assertGreater(len(tr), 0); self.assertGreater(len(te), 0)


class TestPooledLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.te, cls.per = load_all_datasets(project_root, seed=1)

    def test_covers_multiple_datasets(self):
        self.assertGreaterEqual(len(self.per), 3)

    def test_fiw_families_are_disjoint_across_the_pool(self):
        from src.splits import family_of
        fiw_tr = [p for p in self.tr if "FIDs" in p[0]]
        fiw_te = [p for p in self.te if "FIDs" in p[0]]
        if fiw_tr and fiw_te:
            self.assertTrue({family_of(p[0]) for p in fiw_tr}
                            .isdisjoint({family_of(p[0]) for p in fiw_te}))

    def test_both_classes_present_on_each_side(self):
        for side in (self.tr, self.te):
            labels = {p[2] for p in side}
            self.assertEqual(labels, {0, 1})


if __name__ == "__main__":
    unittest.main()


class TestRebuiltNegativesKeepPairs(unittest.TestCase):
    """Splitting KinFaceW on fixed pairs dropped 173/1066 boundary-crossing
    negatives and shrank the test set to 119. Rebuilding negatives per side
    keeps the data while staying group-disjoint."""

    def setUp(self):
        self.pairs = []
        for f in range(30):                       # positives, same family
            self.pairs.append((f"fd_{f:03d}_1.jpg", f"fd_{f:03d}_2.jpg", 1, "fd"))
        for f in range(30):                       # negatives, cross family
            g = (f + 7) % 30
            self.pairs.append((f"fd_{f:03d}_1.jpg", f"fd_{g:03d}_2.jpg", 0, "fd"))

    def test_retains_far_more_pairs_than_naive_split(self):
        from src.multi_dataset import split_rebuild_negatives, _kfw_family
        tr, te = split_rebuild_negatives(self.pairs, _kfw_family, 0.2, seed=0)
        naive_tr, naive_te = _split_by_key(self.pairs, _kfw_family, 0.2, seed=0)
        self.assertGreater(len(tr) + len(te), len(naive_tr) + len(naive_te))

    def test_stays_group_disjoint(self):
        from src.multi_dataset import split_rebuild_negatives, _kfw_family
        tr, te = split_rebuild_negatives(self.pairs, _kfw_family, 0.2, seed=0)
        gt = {_kfw_family(p[0]) for p in tr} | {_kfw_family(p[1]) for p in tr}
        ge = {_kfw_family(p[0]) for p in te} | {_kfw_family(p[1]) for p in te}
        self.assertTrue(gt.isdisjoint(ge))

    def test_each_side_is_balanced(self):
        from src.multi_dataset import split_rebuild_negatives, _kfw_family
        for side in split_rebuild_negatives(self.pairs, _kfw_family, 0.2, seed=0):
            kin = sum(1 for p in side if p[2] == 1)
            non = sum(1 for p in side if p[2] == 0)
            self.assertEqual(kin, non)


class TestGroupKeysAreDatasetScoped(unittest.TestCase):
    """`fd_003` exists in BOTH KinFaceW-I and KinFaceW-II as different
    families. An unscoped key collides them, which corrupts any pooled
    disjointness check."""

    def test_same_family_name_in_different_datasets_is_distinct(self):
        from src.multi_dataset import dataset_group_key
        a = os.path.join("KinFaceW-I", "images", "father-dau", "fd_003_1.jpg")
        b = os.path.join("KinFaceW-II", "images", "father-dau", "fd_003_1.jpg")
        self.assertNotEqual(dataset_group_key(a), dataset_group_key(b))

    def test_same_family_within_one_dataset_matches(self):
        from src.multi_dataset import dataset_group_key
        a = os.path.join("KinFaceW-I", "images", "father-dau", "fd_003_1.jpg")
        b = os.path.join("KinFaceW-I", "images", "father-dau", "fd_003_2.jpg")
        self.assertEqual(dataset_group_key(a), dataset_group_key(b))

    def test_distinguishes_all_four_datasets(self):
        from src.multi_dataset import dataset_group_key
        keys = {
            dataset_group_key(os.path.join("public", "FIDs", "F0001", "MID1", "x.jpg")),
            dataset_group_key(os.path.join("KinFaceW-I", "images", "fd", "fd_003_1.jpg")),
            dataset_group_key(os.path.join("KinFaceW-II", "images", "fd", "fd_003_1.jpg")),
            dataset_group_key(os.path.join("TSKinFace_Data", "x", "FMS-3-F.jpg")),
        }
        self.assertEqual(len(keys), 4)

    def test_pooled_split_is_disjoint_under_scoped_keys(self):
        from src.multi_dataset import load_all_datasets, dataset_group_key
        tr, te, _ = load_all_datasets(project_root, 0.2, seed=3)
        gt = {dataset_group_key(p[0]) for p in tr} | {dataset_group_key(p[1]) for p in tr}
        ge = {dataset_group_key(p[0]) for p in te} | {dataset_group_key(p[1]) for p in te}
        self.assertEqual(len(gt & ge), 0, "real leakage between train and test")
