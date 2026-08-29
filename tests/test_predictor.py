"""The deployment predictor must load a checkpoint and score a pair
reproducibly, with a calibrated threshold rather than a bare 0.5."""
import os, sys, tempfile, unittest, torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictor import KinshipPredictor, RELATIONS


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.models_hybrid import QuantumAugmentedKinshipClassifier
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls._tmp.name, "ckpt.pt")
        torch.save({
            "state_dict": QuantumAugmentedKinshipClassifier().state_dict(),
            "threshold": 0.42,
            "use_quantum": True,
            "metrics": {"accuracy": 77.0, "roc_auc": 0.85},
        }, cls.ckpt)

    def test_loads_checkpoint_and_reports_threshold(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        self.assertAlmostEqual(p.threshold, 0.42, places=6)

    def test_scores_embedding_pair_to_probability_and_decision(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        r = p.predict_embeddings(torch.randn(512), torch.randn(512), "fd")
        self.assertIn("probability", r)
        self.assertIn("is_kin", r)
        self.assertTrue(0.0 <= r["probability"] <= 1.0)
        self.assertEqual(r["is_kin"], r["probability"] >= p.threshold)

    def test_decision_uses_calibrated_threshold_not_half(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        p.threshold = 0.9
        r = p.predict_embeddings(torch.zeros(512), torch.zeros(512), "fd")
        if 0.5 <= r["probability"] < 0.9:
            self.assertFalse(r["is_kin"], "used 0.5 instead of the threshold")

    def test_rejects_unknown_relation(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        with self.assertRaises(ValueError):
            p.predict_embeddings(torch.randn(512), torch.randn(512), "cousin")

    def test_all_declared_relations_are_accepted(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        for rel in RELATIONS:
            r = p.predict_embeddings(torch.randn(512), torch.randn(512), rel)
            self.assertTrue(0.0 <= r["probability"] <= 1.0)

    def test_is_symmetric_in_argument_order(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        a, b = torch.randn(512), torch.randn(512)
        r1 = p.predict_embeddings(a, b, "ms")["probability"]
        r2 = p.predict_embeddings(b, a, "ms")["probability"]
        self.assertAlmostEqual(r1, r2, places=4)

    def test_batch_matches_single(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        a, b = torch.randn(3, 512), torch.randn(3, 512)
        batch = p.predict_batch(a, b, ["fd", "fs", "md"])
        for i, rel in enumerate(["fd", "fs", "md"]):
            one = p.predict_embeddings(a[i], b[i], rel)["probability"]
            self.assertAlmostEqual(batch[i]["probability"], one, places=4)


if __name__ == "__main__":
    unittest.main()


class TestThresholdCalibration(unittest.TestCase):
    """Youden's J weights TPR and FPR equally, which gave a 24.3% false-positive
    rate on held-out FIW. Calibration must support an explicit operating point."""

    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(0)
        # Separable-but-overlapping scores, like the real model produces.
        self.y = np.r_[np.ones(500), np.zeros(500)]
        self.s = np.r_[rng.beta(5, 2, 500), rng.beta(2, 5, 500)]

    def test_accuracy_objective_beats_youden_on_accuracy(self):
        from src.calibration import calibrate_threshold
        from sklearn.metrics import accuracy_score
        t_acc = calibrate_threshold(self.y, self.s, objective="accuracy")
        t_j = calibrate_threshold(self.y, self.s, objective="youden")
        a_acc = accuracy_score(self.y, (self.s >= t_acc).astype(int))
        a_j = accuracy_score(self.y, (self.s >= t_j).astype(int))
        self.assertGreaterEqual(a_acc, a_j)

    def test_max_fpr_constraint_is_respected(self):
        from src.calibration import calibrate_threshold
        t = calibrate_threshold(self.y, self.s, objective="accuracy", max_fpr=0.10)
        fpr = ((self.s >= t) & (self.y == 0)).sum() / (self.y == 0).sum()
        self.assertLessEqual(fpr, 0.10 + 1e-9)

    def test_returns_a_threshold_within_score_range(self):
        from src.calibration import calibrate_threshold
        t = calibrate_threshold(self.y, self.s, objective="accuracy")
        self.assertGreaterEqual(t, self.s.min())
        self.assertLessEqual(t, self.s.max() + 1e-9)

    def test_impossible_fpr_constraint_still_returns_usable_threshold(self):
        from src.calibration import calibrate_threshold
        t = calibrate_threshold(self.y, self.s, objective="accuracy", max_fpr=0.0)
        self.assertTrue(np.isfinite(t))
        fpr = ((self.s >= t) & (self.y == 0)).sum() / (self.y == 0).sum()
        self.assertLessEqual(fpr, 0.02)


import numpy as np  # noqa: E402  (used by the class above)


class TestPerDomainThresholds(unittest.TestCase):
    """One global threshold cost up to 5.9 points across datasets. The
    predictor must support a per-domain operating point."""

    @classmethod
    def setUpClass(cls):
        from src.models_hybrid import QuantumAugmentedKinshipClassifier
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls._tmp.name, "ckpt_dom.pt")
        torch.save({
            "state_dict": QuantumAugmentedKinshipClassifier().state_dict(),
            "threshold": 0.5,
            "domain_thresholds": {"fiw": 0.2, "kinfacew": 0.8},
            "use_quantum": True,
        }, cls.ckpt)

    def test_uses_domain_threshold_when_named(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        a, b = torch.randn(512), torch.randn(512)
        lo = p.predict_embeddings(a, b, "fd", domain="fiw")
        hi = p.predict_embeddings(a, b, "fd", domain="kinfacew")
        self.assertEqual(lo["threshold"], 0.2)
        self.assertEqual(hi["threshold"], 0.8)
        self.assertAlmostEqual(lo["probability"], hi["probability"], places=5)

    def test_unknown_domain_falls_back_to_global(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        r = p.predict_embeddings(torch.randn(512), torch.randn(512), "fd",
                                 domain="does-not-exist")
        self.assertEqual(r["threshold"], 0.5)

    def test_no_domain_uses_global(self):
        p = KinshipPredictor(self.ckpt, device="cpu")
        r = p.predict_embeddings(torch.randn(512), torch.randn(512), "fd")
        self.assertEqual(r["threshold"], 0.5)


class TestThresholdStability(unittest.TestCase):
    """Accuracy varied by up to 6.8 points across seeds while AUC moved only
    0.038 -- the instability was in threshold selection, not the model.
    Picking the single best-scoring cut overfits the validation sample; a
    smoothed choice must be steadier under resampling."""

    def _sample(self, seed):
        rng = np.random.default_rng(seed)
        y = np.r_[np.ones(400), np.zeros(400)]
        s = np.r_[rng.beta(5, 2, 400), rng.beta(2, 5, 400)]
        return y, s

    def test_smoothed_threshold_varies_less_across_resamples(self):
        from src.calibration import calibrate_threshold
        sharp, smooth = [], []
        for seed in range(12):
            y, s = self._sample(seed)
            sharp.append(calibrate_threshold(y, s, objective="accuracy"))
            smooth.append(calibrate_threshold(y, s, objective="accuracy",
                                              smooth=True))
        self.assertLess(np.std(smooth), np.std(sharp),
                        "smoothing did not reduce threshold variance")

    def test_smoothed_threshold_is_still_accurate(self):
        from src.calibration import calibrate_threshold
        from sklearn.metrics import accuracy_score
        y, s = self._sample(99)
        t_sharp = calibrate_threshold(y, s, objective="accuracy")
        t_smooth = calibrate_threshold(y, s, objective="accuracy", smooth=True)
        a_sharp = accuracy_score(y, (s >= t_sharp).astype(int))
        a_smooth = accuracy_score(y, (s >= t_smooth).astype(int))
        self.assertGreater(a_smooth, a_sharp - 0.03,
                           "smoothing cost too much accuracy")
