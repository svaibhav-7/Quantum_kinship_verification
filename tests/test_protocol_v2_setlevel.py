import os
import json
import tempfile
import unittest

import numpy as np

from scripts.evaluation.run_protocol_v2 import (
    build_side,
    galleries,
    identity_of_path,
    Log,
    negative_seed,
    norm,
    set_matrix,
    source_photo,
)


class TestSetLevelNoShared(unittest.TestCase):
    def test_gallery_includes_cached_photos_not_sampled_in_pairs(self):
        with tempfile.TemporaryDirectory() as root:
            first_dir = os.path.join(root, "FIDs", "F0001", "MID1")
            second_dir = os.path.join(root, "FIDs", "F0001", "MID2")
            os.makedirs(first_dir)
            os.makedirs(second_dir)
            paths = [
                os.path.join(first_dir, "P0001_face1.jpg"),
                os.path.join(first_dir, "P0002_face1.jpg"),
                os.path.join(second_dir, "P0001_face2.jpg"),
                os.path.join(second_dir, "P0003_face1.jpg"),
            ]
            cache = {}
            for path in paths:
                with open(path, "wb"):
                    pass
                cache[norm(path)] = np.ones(8, dtype=np.float32)

            got = galleries([(paths[0], paths[2], 1, "fd")], cache)

            self.assertEqual(len(got[identity_of_path(paths[0])]), 2)
            self.assertEqual(len(got[identity_of_path(paths[2])]), 2)

    def test_noshared_set_and_single_arms_use_the_same_query_rows(self):
        first = [
            os.path.join("FIDs", "F0001", "MID1", "P0001_face1.jpg"),
            os.path.join("FIDs", "F0001", "MID1", "P0002_face1.jpg"),
        ]
        second = [
            os.path.join("FIDs", "F0001", "MID2", "P0001_face2.jpg"),
            os.path.join("FIDs", "F0001", "MID2", "P0003_face1.jpg"),
        ]
        # A non-shared query for this identity pair comes first and populates
        # the cached set features; the later shared-source query must still be
        # removed from both arms.
        pairs = [(first[1], second[1], 0, "fd"),
                 (first[0], second[0], 1, "fd")]
        gallery = {
            identity_of_path(first[0]): set(first),
            identity_of_path(second[0]): set(second),
        }
        cache = {
            norm(path): np.arange(8, dtype=np.float32) + i
            for i, path in enumerate(first + second, 1)
        }

        _x_set, _y_set, kept_set = set_matrix(pairs, gallery, cache, "set_noshared")
        _x_single, _y_single, kept_single = set_matrix(
            pairs, gallery, cache, "single_noshared"
        )

        np.testing.assert_array_equal(kept_set, [0])
        np.testing.assert_array_equal(kept_single, kept_set)
        self.assertEqual(source_photo(first[0]), source_photo(second[0]))


class TestNegativeSeedOffset(unittest.TestCase):
    def test_offset_changes_negative_draw_without_changing_positive_rows(self):
        from unittest import mock
        from scripts.evaluation import run_protocol_v2

        positives = [
            (f"family-{i}/father.jpg", f"family-{i}/son.jpg", 1, "fs")
            for i in range(20)
        ]
        family = lambda path: path.split("/")[0]
        with mock.patch.object(run_protocol_v2, "NEGATIVE_SEED_OFFSET", 0):
            first = build_side(positives, family, negative_seed(42))
        with mock.patch.object(run_protocol_v2, "NEGATIVE_SEED_OFFSET", 1000):
            second = build_side(positives, family, negative_seed(42))

        self.assertEqual([row for row in first if row[2] == 1],
                         [row for row in second if row[2] == 1])
        self.assertNotEqual([row for row in first if row[2] == 0],
                            [row for row in second if row[2] == 0])
        self.assertEqual(run_protocol_v2.SPLIT_SEED, 42)

    def test_log_refuses_to_resume_different_protocol_sources(self):
        from unittest import mock
        from scripts.evaluation import run_protocol_v2

        with tempfile.TemporaryDirectory() as output:
            with open(os.path.join(output, "reference_facenet.jsonl"),
                      "w", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "key": "cosine|FIW|f0|s0",
                    "negative_seed_offset": 0,
                    "provenance": {"source_sha256": "stale"},
                }) + "\n")
            with mock.patch.object(run_protocol_v2, "OUT", output):
                with self.assertRaisesRegex(ValueError, "different protocol source"):
                    Log("reference_facenet")


if __name__ == "__main__":
    unittest.main()
