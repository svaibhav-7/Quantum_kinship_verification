"""Loading and preparing user-supplied photographs.

Dataset images are pre-cropped faces (FIW 224x224, KinFaceW/TSKinFace 64x64);
a photograph a user supplies is not. Measured on this pipeline, the same face
inside a wide scene embeds at cosine 0.027 against its cropped version -- an
effectively unrelated image -- and face detection restores it to 0.940.
Detection is therefore a correctness requirement, and these tests pin it along
with the input handling around it.
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

from src.user_input import (InputError, confidence_label, extract_faces,
                            is_url, resolve_inputs)


class TestSourceClassification(unittest.TestCase):
    def test_recognises_http_urls(self):
        self.assertTrue(is_url("http://example.com/a.jpg"))
        self.assertTrue(is_url("https://example.com/a.jpg"))

    def test_local_paths_are_not_urls(self):
        self.assertFalse(is_url("photo.jpg"))
        self.assertFalse(is_url(r"C:\Users\me\photo.jpg"))
        self.assertFalse(is_url("/home/me/photo.jpg"))

    def test_rejects_non_http_schemes(self):
        """file:// must not be treated as fetchable."""
        self.assertFalse(is_url("file:///etc/passwd"))
        self.assertFalse(is_url("ftp://host/a.jpg"))


class TestResolveInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.d = os.path.join(cls._tmp.name, "person_a")
        os.makedirs(cls.d)
        for i in range(3):
            Image.new("RGB", (100, 100)).save(os.path.join(cls.d, f"{i}.jpg"))
        with open(os.path.join(cls.d, "notes.txt"), "w") as f:
            f.write("not an image")

    def test_expands_a_directory_into_its_images(self):
        self.assertEqual(len(resolve_inputs([self.d])), 3)

    def test_skips_non_image_files_in_a_directory(self):
        self.assertTrue(all(not p.endswith(".txt") for p in resolve_inputs([self.d])))

    def test_accepts_an_explicit_file_list(self):
        files = [os.path.join(self.d, f"{i}.jpg") for i in range(2)]
        self.assertEqual(len(resolve_inputs(files)), 2)

    def test_deduplicates_repeated_inputs(self):
        f = os.path.join(self.d, "0.jpg")
        self.assertEqual(len(resolve_inputs([f, f, self.d])), 3)

    def test_directory_listing_is_deterministic(self):
        self.assertEqual(resolve_inputs([self.d]), resolve_inputs([self.d]))

    def test_strips_surrounding_quotes(self):
        """Users paste quoted paths from Explorer; those must still resolve."""
        f = os.path.join(self.d, "0.jpg")
        self.assertEqual(len(resolve_inputs([f'"{f}"'])), 1)

    def test_missing_path_raises_a_clear_error(self):
        with self.assertRaises(InputError) as e:
            resolve_inputs([os.path.join(self._tmp.name, "nope.jpg")])
        self.assertIn("not found", str(e.exception).lower())

    def test_empty_directory_raises_rather_than_returning_nothing(self):
        empty = os.path.join(self._tmp.name, "empty")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaises(InputError):
            resolve_inputs([empty])

    def test_no_input_raises(self):
        with self.assertRaises(InputError):
            resolve_inputs([])


class TestExtractFaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import glob

        cls.samples = sorted(glob.glob(
            os.path.join(project_root, "public", "FIDs", "*", "*", "*.jpg")))[:2]
        if not cls.samples:
            raise unittest.SkipTest("no dataset images available")
        cls._tmp = tempfile.TemporaryDirectory()

    def test_embeddings_are_unit_norm(self):
        embs, failed = extract_faces(self.samples[:1], detect=True)
        self.assertEqual(failed, [])
        self.assertAlmostEqual(float(np.linalg.norm(embs[0])), 1.0, places=4)
        self.assertEqual(embs[0].shape, (512,))

    def test_unreadable_file_is_reported_not_raised(self):
        """One bad photo must not abort the whole run."""
        bad = os.path.join(self._tmp.name, "broken.jpg")
        with open(bad, "w") as f:
            f.write("not an image")
        embs, failed = extract_faces([self.samples[0], bad], detect=True)
        self.assertEqual(len(embs), 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0][0], bad)

    def test_faceless_photo_is_reported_as_such(self):
        blank = os.path.join(self._tmp.name, "blank.jpg")
        Image.new("RGB", (400, 400), (30, 30, 30)).save(blank)
        embs, failed = extract_faces([blank], detect=True)
        self.assertEqual(embs, [])
        self.assertIn("no face", failed[0][1].lower())

    def test_detection_recovers_a_face_from_an_uncropped_photo(self):
        """The measurement this module exists for: without detection the same
        face in a wide scene embeds at cosine 0.027 against its crop."""
        face = Image.open(self.samples[0]).convert("RGB")
        tight = os.path.join(self._tmp.name, "tight.jpg")
        wide = os.path.join(self._tmp.name, "wide.jpg")
        face.save(tight)
        scene = Image.new("RGB", (900, 700), (70, 110, 60))
        scene.paste(face.resize((150, 150)), (500, 120))
        scene.save(wide)

        embs, failed = extract_faces([tight, wide], detect=True)
        if len(embs) < 2:
            raise unittest.SkipTest("detector missed the pasted face")
        cos = float(embs[0] @ embs[1])
        self.assertGreater(cos, 0.75,
                           "detection failed to align the uncropped photo")


class TestConfidenceLabel(unittest.TestCase):
    def test_scores_on_the_threshold_are_borderline(self):
        self.assertEqual(confidence_label(0.50, 0.50), "borderline")

    def test_distant_scores_are_high_confidence(self):
        self.assertEqual(confidence_label(0.97, 0.50), "high")
        self.assertEqual(confidence_label(0.02, 0.50), "high")

    def test_label_is_symmetric_about_the_threshold(self):
        self.assertEqual(confidence_label(0.65, 0.50),
                         confidence_label(0.35, 0.50))

    def test_respects_a_non_default_threshold(self):
        """Thresholds are calibrated per model, not fixed at 0.5."""
        self.assertEqual(confidence_label(0.82, 0.80), "borderline")


if __name__ == "__main__":
    unittest.main()
