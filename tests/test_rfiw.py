"""Parsing the official RFIW Track-I files without changing their splits."""
import os
import tempfile
import unittest

from src.rfiw import load_identity_pairs, load_labelled_pairs


def touch(root, *relative):
    for path in relative:
        full = os.path.join(root, *path.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").close()


class TestLoadLabelledPairs(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.csv = os.path.join(self.dir.name, "test-pairs.csv")
        with open(self.csv, "w") as stream:
            stream.write("labels,p1,p2,ptype\n"
                         "1,F1/MID1/a.jpg,F1/MID2/b.jpg,fs\n"
                         "0,F1/MID1/a.jpg,F2/MID1/c.jpg,fs\n"
                         "1,F3/MID1/gone.jpg,F3/MID2/d.jpg,md\n")
        self.faces = os.path.join(self.dir.name, "faces")
        touch(self.faces, "F1/MID1/a.jpg", "F1/MID2/b.jpg", "F2/MID1/c.jpg",
              "F3/MID2/d.jpg")

    def test_keeps_official_labels(self):
        pairs = load_labelled_pairs(self.csv, self.faces)
        self.assertEqual([p[2] for p in pairs], [1, 0])

    def test_skips_rows_with_a_missing_face(self):
        pairs = load_labelled_pairs(self.csv, self.faces)
        self.assertFalse(any("gone" in p[0] for p in pairs))

    def test_paths_resolve_on_disk(self):
        for first, second, _label, relation in load_labelled_pairs(self.csv, self.faces):
            self.assertTrue(os.path.isfile(first))
            self.assertTrue(os.path.isfile(second))
            self.assertEqual(relation, "fs")

    def test_relation_filter(self):
        pairs = load_labelled_pairs(self.csv, self.faces, relations={"md"})
        self.assertEqual(pairs, [])


class TestLoadIdentityPairs(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.csv = os.path.join(self.dir.name, "train-pairs.csv")
        with open(self.csv, "w") as stream:
            stream.write("p1,p2,ptype,fid1,fid2\n"
                         "F1/MID1,F1/MID2,fs,F1,F1\n"
                         "F1/MID1,F1/MID3,bb,F1,F1\n"
                         "F2/MID1,F2/MID9,md,F2,F2\n")
        self.faces = os.path.join(self.dir.name, "faces")
        touch(self.faces, "F1/MID1/a.jpg", "F1/MID1/b.jpg",
              "F1/MID2/c.jpg", "F1/MID2/d.jpg", "F1/MID3/e.jpg",
              "F2/MID1/f.jpg")

    def test_expands_identities_to_photo_pairs(self):
        pairs = load_identity_pairs(self.csv, self.faces, relations={"fs"})
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(p[2] == 1 and p[3] == "fs" for p in pairs))

    def test_identity_without_photos_yields_nothing(self):
        pairs = load_identity_pairs(self.csv, self.faces, relations={"md"})
        self.assertEqual(pairs, [])

    def test_cap_limits_each_identity_pair_and_is_deterministic(self):
        first = load_identity_pairs(self.csv, self.faces, relations={"fs"},
                                    cap_per_identity_pair=2, seed=5)
        second = load_identity_pairs(self.csv, self.faces, relations={"fs"},
                                     cap_per_identity_pair=2, seed=5)
        self.assertEqual(len(first), 2)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
