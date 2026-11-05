"""Pooling scores across folds without letting fold scale decide the result."""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.evaluation import summarize_protocol_v2
from src.fair_protocol import pool_fold_scores


class TestPoolFoldScores(unittest.TestCase):
    def test_fold_score_scale_cannot_change_the_pooled_result(self):
        # Each fold separates its classes perfectly, but fold B's scores are all
        # higher. Raw pooling would mix them; ranking within a fold must not.
        fold_a = {"scores": [np.array([0.1, 0.2, 0.3, 0.4])],
                  "y": np.array([0, 0, 1, 1]),
                  "groups_a": np.array(list("aabb")),
                  "groups_b": np.array(list("eeff"))}
        fold_b = {"scores": [np.array([10.0, 20.0, 30.0, 40.0])],
                  "y": np.array([0, 0, 1, 1]),
                  "groups_a": np.array(list("ccdd")),
                  "groups_b": np.array(list("gghh"))}
        y, scores, groups_a, groups_b = pool_fold_scores([fold_a, fold_b])
        self.assertEqual(roc_auc_score(y, scores), 1.0)
        self.assertEqual(list(groups_a), list("aabbccdd"))
        self.assertEqual(list(groups_b), list("eeffgghh"))

    def test_seeds_within_a_fold_are_averaged(self):
        fold = {"scores": [np.array([1.0, 2.0]), np.array([2.0, 1.0])],
                "y": np.array([0, 1]),
                "groups_a": np.array(list("ab")),
                "groups_b": np.array(list("cd"))}
        _y, scores, _groups_a, _groups_b = pool_fold_scores([fold])
        self.assertEqual(scores[0], scores[1])


class TestLoadFolds(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.scores_dir = os.path.join(self.tempdir.name, "scores")
        os.makedirs(self.scores_dir)
        self.out_patch = mock.patch.object(summarize_protocol_v2, "OUT", self.tempdir.name)
        self.out_patch.start()
        self.addCleanup(self.out_patch.stop)
        self.folds_patch = mock.patch.object(summarize_protocol_v2, "FOLDS", 1)
        self.folds_patch.start()
        self.addCleanup(self.folds_patch.stop)
        summarize_protocol_v2.INCOMPLETE_COMPARISONS.clear()
        self.addCleanup(summarize_protocol_v2.INCOMPLETE_COMPARISONS.clear)

    def write_scores(self, seed, **arrays):
        np.savez_compressed(
            os.path.join(self.scores_dir, f"reference_facenet_fair_FIW_f0_s{seed}.npz"),
            **arrays)

    def test_legacy_scores_are_reported_incomplete(self):
        self.write_scores(0, matched=np.array([0.1, 0.9]), y=np.array([0, 1]))

        loaded = summarize_protocol_v2.load_folds(
            "reference", "facenet", "fair", "FIW", "matched")

        self.assertIsNone(loaded)
        self.assertIn("legacy or incomplete score schema",
                      summarize_protocol_v2.INCOMPLETE_COMPARISONS[
                          ("reference", "facenet", "fair", "FIW", "matched", 0)])

    def test_seed_files_must_have_identical_pair_rows(self):
        base = {
            "matched": np.array([0.1, 0.9]),
            "y": np.array([0, 1]),
            "groups_a": np.array(["family-a", "family-b"]),
            "groups_b": np.array(["family-c", "family-d"]),
        }
        self.write_scores(0, **base)
        self.write_scores(1, **{**base, "y": np.array([1, 0])})

        loaded = summarize_protocol_v2.load_folds(
            "reference", "facenet", "fair", "FIW", "matched")

        self.assertIsNone(loaded)
        self.assertIn("disagree on pair labels or family IDs",
                      summarize_protocol_v2.INCOMPLETE_COMPARISONS[
                          ("reference", "facenet", "fair", "FIW", "matched", 0)])

    def test_negative_seed_offsets_are_read_from_run_logs(self):
        path = os.path.join(self.tempdir.name, "reference_facenet.jsonl")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write('{"negative_seed_offset": 1000}\n')
            stream.write('{"negative_seed_offset": 2000}\n')

        self.assertEqual(summarize_protocol_v2.negative_seed_offsets(), [1000, 2000])


class TestAlign(unittest.TestCase):
    def test_indexed_arms_keep_the_same_intersection_in_order(self):
        first = [{
            "scores": [np.array([0.1, 0.2, 0.3])],
            "y": np.array([0, 1, 0]),
            "groups_a": np.array(["a", "b", "c"]),
            "groups_b": np.array(["x", "y", "z"]),
            "index": np.array([0, 1, 2]),
        }]
        second = [{
            "scores": [np.array([0.8, 0.9])],
            "y": np.array([1, 0]),
            "groups_a": np.array(["b", "c"]),
            "groups_b": np.array(["y", "z"]),
            "index": np.array([1, 2]),
        }]

        aligned_first, aligned_second = summarize_protocol_v2.align(first, second)

        np.testing.assert_array_equal(aligned_first[0]["scores"][0], [0.2, 0.3])
        np.testing.assert_array_equal(aligned_first[0]["y"], [1, 0])
        np.testing.assert_array_equal(aligned_second[0]["scores"][0], [0.8, 0.9])


class TestTrainingRunSummary(unittest.TestCase):
    def test_model_runs_repeated_across_corpora_are_counted_once(self):
        records = [
            {"arm": "fair", "corpus": corpus, "fold": 2, "seed": 1,
             "val_auc": 0.72, "val_cosine_auc": 0.75, "train_auc": 0.88,
             "failed": True, "epochs": 9}
            for corpus in ("FIW", "TSKinFace")
        ]
        records.append({"arm": "cosine", "corpus": "FIW", "fold": 2, "seed": 0,
                        "failed": False})

        runs = summarize_protocol_v2.training_run_summary(records)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["arm"], "fair")
        self.assertEqual(runs[0]["train_auc"], 0.88)

    def test_conflicting_model_metadata_is_rejected(self):
        records = [
            {"arm": "fair", "corpus": corpus, "fold": 0, "seed": 0,
             "val_auc": value}
            for corpus, value in (("FIW", 0.7), ("TSKinFace", 0.8))
        ]
        with self.assertRaisesRegex(ValueError, "Inconsistent val_auc"):
            summarize_protocol_v2.training_run_summary(records)


if __name__ == "__main__":
    unittest.main()
