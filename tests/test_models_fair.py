"""The protocol-v2 reference model: a cosine prior plus a learned projection."""
import unittest

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.models_fair import (FairKinshipModel, RelationFreeHybrid,
                             ScalarReadoutModel)


def synthetic_pairs(n, seed, signal_dims=8, dim=64):
    """Kin share a vector only in a few dimensions; the rest is distractor.

    Plain cosine is diluted by the distractor dimensions, so a model that learns
    the projection must beat it.
    """
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, dim, generator=gen)
    b = torch.randn(n, dim, generator=gen)
    y = torch.randint(0, 2, (n,), generator=gen).float()
    kin = y.bool()
    b[kin, :signal_dims] = a[kin, :signal_dims]
    return a, b, y


class TestFairKinshipModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = FairKinshipModel(embed_dim=64, proj_dim=16, dropout=0.0).eval()
        self.a = torch.randn(32, 64)
        self.b = torch.randn(32, 64)

    def test_argument_order_does_not_change_the_score(self):
        forward = self.model.forward_logits(self.a, self.b)
        backward = self.model.forward_logits(self.b, self.a)
        self.assertTrue(torch.allclose(forward, backward))

    def test_embedding_scale_does_not_change_the_score(self):
        # The old model added a learned vector to unit-norm embeddings, which
        # made it depend on the embedding scale and collapse to chance.
        base = self.model.forward_logits(self.a, self.b)
        scaled = self.model.forward_logits(self.a * 7.0, self.b * 0.3)
        self.assertTrue(torch.allclose(base, scaled, atol=1e-5))

    def test_identical_faces_outscore_unrelated_faces_before_training(self):
        same = self.model.forward_logits(self.a, self.a).mean()
        different = self.model.forward_logits(self.a, self.b).mean()
        self.assertGreater(float(same), float(different))

    def test_output_shape(self):
        self.assertEqual(self.model.forward_logits(self.a, self.b).shape, (32, 1))

    def test_training_beats_plain_cosine_on_held_out_data(self):
        a, b, y = synthetic_pairs(3000, seed=1)
        ta, tb, ty = synthetic_pairs(1000, seed=2)
        cosine = torch.nn.functional.cosine_similarity(ta, tb, dim=1)
        baseline = roc_auc_score(ty.numpy(), cosine.numpy())

        model = FairKinshipModel(embed_dim=64, proj_dim=16, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        loss = torch.nn.BCEWithLogitsLoss()
        for _ in range(250):
            optimizer.zero_grad()
            loss(model.forward_logits(a, b).view(-1), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = model.forward_logits(ta, tb).view(-1).numpy()
        self.assertGreater(roc_auc_score(ty.numpy(), scores), baseline + 0.03)


class TestRelationFreeHybrid(unittest.TestCase):
    """The SWAP-branch host model, without the path that collapsed training."""

    def setUp(self):
        torch.manual_seed(0)
        self.a = torch.nn.functional.normalize(torch.randn(16, 512), dim=1)
        self.b = torch.nn.functional.normalize(torch.randn(16, 512), dim=1)
        self.zero = torch.zeros(16, 4)

    def test_encoder_sees_the_face_with_no_added_vector(self):
        model = RelationFreeHybrid(dropout=0.0, use_quantum=False).eval()
        h1, _h2 = model._encode(self.a, self.b, self.zero)
        self.assertTrue(torch.allclose(h1, model.encoder(self.a)))

    def test_relation_labels_cannot_change_the_score(self):
        model = RelationFreeHybrid(dropout=0.0, use_quantum=False).eval()
        other = torch.eye(4)[torch.randint(0, 4, (16,))]
        self.assertTrue(torch.allclose(model.forward_logits(self.a, self.b, self.zero),
                                       model.forward_logits(self.a, self.b, other)))

    def test_quantum_switch_changes_predictions(self):
        on = RelationFreeHybrid(dropout=0.0, use_quantum=True).eval()
        off = RelationFreeHybrid(dropout=0.0, use_quantum=False).eval()
        off.load_state_dict(on.state_dict())
        self.assertFalse(torch.allclose(on.forward_logits(self.a, self.b, self.zero),
                                        off.forward_logits(self.a, self.b, self.zero)))

    def test_gradients_reach_the_quantum_parameters(self):
        model = RelationFreeHybrid(dropout=0.0, use_quantum=True)
        model.forward_logits(self.a, self.b, self.zero).sum().backward()
        self.assertGreater(float(model.ent_params1.grad.abs().sum()), 0.0)


class TestScalarReadoutModel(unittest.TestCase):
    """A fair scalar arm: the raw |0...0> probability is tiny and is not a
    similarity, so it cannot alone decide whether one number is enough."""

    def setUp(self):
        torch.manual_seed(0)
        self.a = torch.randn(64, 512)
        self.b = torch.randn(64, 512)

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            ScalarReadoutModel(kind="nonsense")

    def test_swap_scalar_is_the_squared_overlap_of_the_projections(self):
        model = ScalarReadoutModel(kind="swap", n_qubits_each=3, depth=1,
                                   dropout=0.0).eval()
        x = torch.nn.functional.normalize(model.proj(self.a), dim=1)
        y = torch.nn.functional.normalize(model.proj(self.b), dim=1)
        expected = ((x * y).sum(1, keepdim=True)) ** 2
        self.assertTrue(torch.allclose(model.raw_scalar(self.a, self.b), expected,
                                       atol=1e-5))

    def test_swap_scalar_is_order_invariant(self):
        model = ScalarReadoutModel(kind="swap", n_qubits_each=3, depth=1,
                                   dropout=0.0).eval()
        self.assertTrue(torch.allclose(model.forward_logits(self.a, self.b),
                                       model.forward_logits(self.b, self.a),
                                       atol=1e-5))

    def test_log_fidelity_feature_is_rescaled_to_unit_variance(self):
        model = ScalarReadoutModel(kind="fidelity_log", n_qubits_each=3, depth=1,
                                   dropout=0.0).train()
        feature = model.feature(self.a, self.b)
        self.assertAlmostEqual(float(feature.std()), 1.0, delta=0.1)

    def test_predict_averages_both_orders(self):
        model = ScalarReadoutModel(kind="fidelity_log", n_qubits_each=3, depth=1,
                                   dropout=0.0).eval()
        expected = 0.5 * (torch.sigmoid(model.forward_logits(self.a, self.b))
                          + torch.sigmoid(model.forward_logits(self.b, self.a)))
        got = model.predict(self.a, self.b)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
