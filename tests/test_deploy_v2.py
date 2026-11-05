"""Deployment must serve the fair model and calibrate on held-out data."""
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from src.models_fair import FairKinshipModel, RelationFreeHybrid
from src.predictor import KinshipPredictor
from src.set_predictor import train_set_model, train_triad_model
from scripts.deploy.package_v2 import scores as package_scores


class TestFairCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls._tmp.name, "fair.pt")
        torch.manual_seed(0)
        cls.model = FairKinshipModel()
        torch.save({"model_type": "fair", "state_dict": cls.model.state_dict(),
                    "threshold": 0.42, "metrics": {"roc_auc": 0.8}}, cls.path)
        cls.a, cls.b = torch.randn(512), torch.randn(512)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_scores_match_the_saved_model(self):
        predictor = KinshipPredictor(self.path, device="cpu")
        self.model.eval()
        with torch.no_grad():
            expected = float(self.model(self.a[None], self.b[None]))
        got = predictor.predict_embeddings(self.a, self.b, "fd")["probability"]
        self.assertAlmostEqual(got, expected, places=5)

    def test_the_relation_cannot_change_the_decision(self):
        predictor = KinshipPredictor(self.path, device="cpu")
        scores = {predictor.predict_embeddings(self.a, self.b, r)["probability"]
                  for r in ("fd", "fs", "md", "ms")}
        self.assertEqual(len(scores), 1)

    def test_decision_uses_the_stored_threshold(self):
        predictor = KinshipPredictor(self.path, device="cpu")
        result = predictor.predict_embeddings(self.a, self.b, "fd")
        self.assertEqual(result["threshold"], 0.42)
        self.assertEqual(result["is_kin"], result["probability"] >= 0.42)

    def test_relation_can_be_omitted(self):
        predictor = KinshipPredictor(self.path, device="cpu")
        result = predictor.predict_embeddings(self.a, self.b, None)
        self.assertIn("probability", result)


class TestRelationFreeHybridCheckpoint(unittest.TestCase):
    def test_selected_hybrid_checkpoint_loads_and_ignores_relation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hybrid.pt")
            model = RelationFreeHybrid(use_quantum=False)
            torch.save({
                "model_type": "relation_free_hybrid",
                "use_quantum": False,
                "state_dict": model.state_dict(),
                "threshold": 0.4,
            }, path)

            predictor = KinshipPredictor(path, device="cpu")
            first_embedding, second_embedding = torch.randn(512), torch.randn(512)
            first = predictor.predict_embeddings(first_embedding, second_embedding, "fd")
            second = predictor.predict_embeddings(first_embedding, second_embedding, "ms")

            self.assertEqual(predictor.model_type, "relation_free_hybrid")
            self.assertEqual(first["threshold"], 0.4)
            self.assertAlmostEqual(first["probability"], second["probability"], places=6)


class TestPackageScores(unittest.TestCase):
    def test_pair_model_scores_remain_probabilities_for_threshold_calibration(self):
        torch.manual_seed(4)
        first, second = torch.randn(4, 512), torch.randn(4, 512)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        model = FairKinshipModel()
        model.eval()

        got = package_scores(model, (first, second, labels), "fair")
        with torch.no_grad():
            expected = model(first, second).view(-1).numpy()

        np.testing.assert_allclose(got, expected)
        self.assertTrue(np.all((got >= 0.0) & (got <= 1.0)))


class HeldOutCalibration(unittest.TestCase):
    """A separable fit set would give an extreme threshold if calibrated on itself."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.rng = np.random.default_rng(0)

    def vector(self):
        v = self.rng.normal(size=16)
        return v / np.linalg.norm(v)

    def related(self, base, noise):
        v = base + self.rng.normal(0, noise, 16)
        return v / np.linalg.norm(v)


class TestSetModelCalibration(HeldOutCalibration):
    def make(self, noise, n=80):
        galleries, pairs = {}, []
        for i in range(n):
            base = self.vector()
            galleries[f"p{i}"] = np.stack([self.related(base, noise) for _ in range(3)])
            galleries[f"c{i}"] = np.stack([self.related(base, noise) for _ in range(3)])
            galleries[f"s{i}"] = np.stack([self.vector() for _ in range(3)])
            pairs += [(f"p{i}", f"c{i}", 1), (f"p{i}", f"s{i}", 0)]
        return galleries, pairs

    def test_threshold_comes_from_validation_when_given(self):
        fit = self.make(0.02)
        val = self.make(0.45)
        path = os.path.join(self.dir.name, "m.pkl")
        train_set_model(*fit, path)
        self_calibrated = pickle.load(open(path, "rb"))["threshold"]
        train_set_model(*fit, path, val_galleries=val[0], val_pairs=val[1])
        stored = pickle.load(open(path, "rb"))
        self.assertTrue(stored["calibrated_on_validation"])
        self.assertNotAlmostEqual(stored["threshold"], self_calibrated, places=3)

    def test_without_validation_the_file_says_so(self):
        fit = self.make(0.02)
        path = os.path.join(self.dir.name, "m.pkl")
        train_set_model(*fit, path)
        self.assertFalse(pickle.load(open(path, "rb"))["calibrated_on_validation"])


class TestTriadModelCalibration(HeldOutCalibration):
    def make(self, noise, n=80):
        rows = []
        for _ in range(n):
            base = self.vector()
            f, m = self.related(base, noise), self.related(base, noise)
            rows.append((f, m, self.related(base, noise), 1))
            rows.append((f, m, self.vector(), 0))
        return rows

    def test_threshold_comes_from_validation_when_given(self):
        path = os.path.join(self.dir.name, "t.pkl")
        fit, val = self.make(0.02), self.make(0.45)
        train_triad_model(fit, path)
        self_calibrated = pickle.load(open(path, "rb"))["threshold"]
        train_triad_model(fit, path, val_rows=val)
        stored = pickle.load(open(path, "rb"))
        self.assertTrue(stored["calibrated_on_validation"])
        self.assertNotAlmostEqual(stored["threshold"], self_calibrated, places=3)


if __name__ == "__main__":
    unittest.main()
