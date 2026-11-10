import json
import os
import tempfile
import unittest

from scripts.evaluation.compare_negative_sampling import compare_runs, render_markdown


class TestNegativeSamplingComparison(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def make_run(self, name, offset, delta, source="source-hash"):
        path = os.path.join(self.tempdir.name, name)
        os.makedirs(path)
        row = {
            "backbone": "facenet",
            "corpus": "FIW",
            "a": "fair",
            "b": "cosine",
            "view": "matched",
            "delta": delta,
            "ci_low": delta - 0.1,
            "ci_high": delta + 0.1,
        }
        with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as stream:
            json.dump({"negative_seed_offsets": [offset], "reference": [row]}, stream)
        with open(os.path.join(path, "reference_facenet.jsonl"), "w",
                  encoding="utf-8") as stream:
            stream.write(json.dumps({
                "negative_seed_offset": offset,
                "provenance": {"source_sha256": source},
            }) + "\n")
        return path

    def test_computes_descriptive_across_draw_variation(self):
        first = self.make_run("offset0", 0, 0.1)
        second = self.make_run("offset1000", 1000, 0.3)

        result = compare_runs([first, second])
        row = result["comparisons"][0]

        self.assertEqual(row["n_draws"], 2)
        self.assertAlmostEqual(row["mean_delta"], 0.2)
        self.assertAlmostEqual(row["sd_delta"], 0.1414213562)
        self.assertEqual([draw["negative_seed_offset"] for draw in row["draws"]],
                         [0, 1000])
        self.assertIn("Per-draw family-bootstrap CIs are not pooled",
                      result["interpretation"])
        self.assertIn("SD across draws", render_markdown(result))

    def test_rejects_duplicate_offsets(self):
        first = self.make_run("first", 0, 0.1)
        second = self.make_run("second", 0, 0.2)
        with self.assertRaisesRegex(ValueError, "distinct negative-seed offset"):
            compare_runs([first, second])

    def test_rejects_different_source_fingerprints(self):
        first = self.make_run("first", 0, 0.1, source="source-a")
        second = self.make_run("second", 1000, 0.2, source="source-b")
        with self.assertRaisesRegex(ValueError, "different protocol source"):
            compare_runs([first, second])


if __name__ == "__main__":
    unittest.main()
