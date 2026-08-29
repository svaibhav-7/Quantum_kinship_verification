"""Deployment inference for kinship verification.

Loads a checkpoint saved by scripts/training/train_honest.py and scores face
pairs. The decision uses the threshold calibrated on a family-disjoint
validation set, not a bare 0.5 -- the operating point matters more than the
raw probability, and 0.5 is rarely optimal on this task.
"""

import os

import torch

RELATIONS = ("fd", "fs", "md", "ms")
_REL_INDEX = {r: i for i, r in enumerate(RELATIONS)}


class KinshipPredictor:
    def __init__(self, checkpoint_path, device="auto"):
        from .models_hybrid import QuantumAugmentedKinshipClassifier

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Accept a bare state_dict as well as the richer bundle.
        if "state_dict" in ckpt:
            state, self.threshold = ckpt["state_dict"], float(ckpt.get("threshold", 0.5))
            self.use_quantum = bool(ckpt.get("use_quantum", True))
            self.metrics = ckpt.get("metrics", {})
            # Per-domain operating points: one global threshold cost up to
            # 5.9 accuracy points across datasets.
            self.domain_thresholds = dict(ckpt.get("domain_thresholds", {}))
        else:
            state, self.threshold = ckpt, 0.5
            self.use_quantum, self.metrics = True, {}
            self.domain_thresholds = {}

        self.model = QuantumAugmentedKinshipClassifier(use_quantum=self.use_quantum)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def _relation_tensor(self, relations):
        idx = []
        for r in relations:
            key = str(r).lower()
            if key not in _REL_INDEX:
                raise ValueError(
                    f"unknown relation {r!r}; expected one of {list(RELATIONS)}"
                )
            idx.append(_REL_INDEX[key])
        return torch.eye(4)[idx].to(self.device)

    @torch.no_grad()
    def threshold_for(self, domain=None):
        """Threshold for a named domain, falling back to the global one."""
        if domain is None:
            return self.threshold
        return self.domain_thresholds.get(str(domain).lower(), self.threshold)

    @torch.no_grad()
    def predict_batch(self, emb1, emb2, relations, domain=None):
        e1 = torch.as_tensor(emb1, dtype=torch.float32).to(self.device)
        e2 = torch.as_tensor(emb2, dtype=torch.float32).to(self.device)
        if e1.dim() == 1:
            e1, e2 = e1.unsqueeze(0), e2.unsqueeze(0)
        rels = self._relation_tensor(relations)

        # Kinship is symmetric; average both orders so the answer cannot
        # depend on which face the caller passed first.
        p = 0.5 * (self.model(e1, e2, rels) + self.model(e2, e1, rels))
        p = p.view(-1).cpu()
        thr = self.threshold_for(domain)
        return [
            {
                "probability": float(v),
                "is_kin": bool(v >= thr),
                "threshold": thr,
            }
            for v in p
        ]

    def predict_embeddings(self, emb1, emb2, relation, domain=None):
        return self.predict_batch(emb1, emb2, [relation], domain=domain)[0]

    def predict_images(self, img1, img2, relation, extractor=None, domain=None):
        """Score two image paths. Builds a FaceNet extractor if not supplied."""
        if extractor is None:
            from .models_improved import FaceFeatureExtractor

            extractor = FaceFeatureExtractor()
        e1 = torch.tensor(extractor.extract(img1), dtype=torch.float32)
        e2 = torch.tensor(extractor.extract(img2), dtype=torch.float32)
        return self.predict_embeddings(e1, e2, relation, domain=domain)
