import unittest
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.training.train_with_fiw_improved import split_families


class TestFamilySplit(unittest.TestCase):
    def test_split_families_keeps_disjoint_sets_and_minimum_size(self):
        families = [f"F{i:04d}" for i in range(10)]

        train_families, val_families = split_families(families, val_ratio=0.2, seed=42)

        self.assertTrue(len(train_families) > 0)
        self.assertTrue(len(val_families) > 0)
        self.assertTrue(set(train_families).isdisjoint(set(val_families)))
        self.assertEqual(len(train_families) + len(val_families), len(families))


if __name__ == "__main__":
    unittest.main()
