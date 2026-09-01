"""User-facing verification tool.

Accepts local paths, URLs, and directories; detects and crops faces (user
photos are not pre-cropped like dataset images); scores person-vs-person.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts", "inference"))

from src.user_input import (InputError, is_url, resolve_inputs,
                            confidence_label)


class TestUrlDetection(unittest.TestCase):
    def test_recognises_http_and_https(self):
        self.assertTrue(is_url("http://example.com/a.jpg"))
        self.assertTrue(is_url("https://example.com/a.jpg"))

    def test_local_paths_are_not_urls(self):
        self.assertFalse(is_url("photo.jpg"))
        self.assertFalse(is_url("C:\\photos\\a.jpg"))
        self.assertFalse(is_url("/home/user/a.jpg"))
        self.assertFalse(is_url("./rel/a.jpg"))


class TestResolveInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.d = cls._tmp.name
        cls.files = []
        for i in range(3):
            p = os.path.join(cls.d, f"img{i}.jpg")
            Image.new("RGB", (64, 64), (i * 40, 100, 150)).save(p)
            cls.files.append(p)
        # a non-image that must be ignored when expanding a directory
        with open(os.path.join(cls.d, "notes.txt"), "w") as f:
            f.write("not an image")

    def test_accepts_a_single_file(self):
        self.assertEqual(resolve_inputs([self.files[0]]), [self.files[0]])

    def test_accepts_several_files(self):
        self.assertEqual(len(resolve_inputs(self.files)), 3)

    def test_expands_a_directory_to_its_images(self):
        got = resolve_inputs([self.d])
        self.assertEqual(len(got), 3)
        self.assertTrue(all(p.lower().endswith(".jpg") for p in got))

    def test_ignores_non_image_files_in_a_directory(self):
        self.assertTrue(all("notes.txt" not in p for p in resolve_inputs([self.d])))

    def test_missing_path_raises_a_clear_error(self):
        with self.assertRaises(InputError) as cm:
            resolve_inputs([os.path.join(self.d, "nope.jpg")])
        self.assertIn("not found", str(cm.exception).lower())

    def test_empty_directory_raises_a_clear_error(self):
        empty = os.path.join(self.d, "empty")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(InputError) as cm:
            resolve_inputs([empty])
        self.assertIn("no images", str(cm.exception).lower())

    def test_deduplicates_repeated_inputs(self):
        got = resolve_inputs([self.files[0], self.files[0]])
        self.assertEqual(len(got), 1)


class TestConfidenceLabel(unittest.TestCase):
    """A probability near the threshold must not be reported as a firm answer:
    held-out accuracy is ~73-77%, so roughly one call in four is wrong."""

    def test_far_above_threshold_is_confident(self):
        self.assertEqual(confidence_label(0.95, 0.50), "high")

    def test_far_below_threshold_is_confident(self):
        self.assertEqual(confidence_label(0.05, 0.50), "high")

    def test_near_threshold_is_borderline(self):
        self.assertEqual(confidence_label(0.52, 0.50), "borderline")
        self.assertEqual(confidence_label(0.48, 0.50), "borderline")

    def test_moderate_distance_is_moderate(self):
        self.assertEqual(confidence_label(0.68, 0.50), "moderate")


class TestFaceExtraction(unittest.TestCase):
    def test_reports_when_no_face_is_found(self):
        """A blank image has no face; the tool must say so, not score noise."""
        from src.user_input import extract_faces

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blank.jpg")
            Image.new("RGB", (200, 200), (128, 128, 128)).save(p)
            embs, failed = extract_faces([p], detect=True)
            self.assertEqual(len(embs), 0)
            self.assertEqual(len(failed), 1)
            self.assertIn(p, failed[0][0])

    def test_detect_off_always_returns_an_embedding(self):
        from src.user_input import extract_faces

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blank.jpg")
            Image.new("RGB", (200, 200), (128, 128, 128)).save(p)
            embs, failed = extract_faces([p], detect=False)
            self.assertEqual(len(embs), 1)
            self.assertEqual(len(failed), 0)
            self.assertEqual(embs[0].shape, (512,))


if __name__ == "__main__":
    unittest.main()
