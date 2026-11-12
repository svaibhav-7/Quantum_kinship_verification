"""Evaluation protocol v2: negatives that cannot be spotted without kinship."""
import unittest

import numpy as np

from sklearn.metrics import roc_auc_score

from src.fair_protocol import (
    _weighted_auc_from_plan,
    _weighted_auc_plan,
    adjust_pvalues,
    bootstrap_auc_delta,
    build_side,
    colour_matched_negatives,
    colour_signature,
    drop_duplicate_groups,
    is_failed_run,
    matched_negatives,
    split_fit_val,
    t_interval,
)


def family(path):
    return path.split("/")[0]


def role(path):
    return path.split("/")[1]


def make_positives(n_families=12):
    """Families each with a father-son and a mother-daughter pair."""
    pairs = []
    for i in range(n_families):
        fam = "fam%02d" % i
        pairs.append((f"{fam}/father.jpg", f"{fam}/son.jpg", 1, "fs"))
        pairs.append((f"{fam}/mother.jpg", f"{fam}/daughter.jpg", 1, "md"))
    return pairs


class TestMatchedNegatives(unittest.TestCase):
    def setUp(self):
        self.pos = make_positives()
        self.neg = matched_negatives(self.pos, family, seed=1)

    def test_one_negative_per_positive(self):
        self.assertEqual(len(self.neg), len(self.pos))
        self.assertTrue(all(p[2] == 0 for p in self.neg))

    def test_negatives_never_share_a_family(self):
        for a, b, _l, _r in self.neg:
            self.assertNotEqual(family(a), family(b))

    def test_relation_label_does_not_predict_kinship(self):
        for rel in ("fs", "md"):
            n_pos = sum(p[3] == rel for p in self.pos)
            n_neg = sum(p[3] == rel for p in self.neg)
            self.assertEqual(n_pos, n_neg)

    def test_second_face_keeps_the_generation_and_gender_of_the_relation(self):
        expected_child = {"fs": "son.jpg", "md": "daughter.jpg"}
        for _a, b, _l, rel in self.neg:
            self.assertEqual(role(b), expected_child[rel])

    def test_first_face_is_the_positive_parent(self):
        expected_parent = {"fs": "father.jpg", "md": "mother.jpg"}
        for a, _b, _l, rel in self.neg:
            self.assertEqual(role(a), expected_parent[rel])

    def test_deterministic_for_a_seed(self):
        self.assertEqual(self.neg, matched_negatives(self.pos, family, seed=1))

    def test_relation_present_in_one_family_only_is_dropped_not_faked(self):
        pos = self.pos + [("famX/father.jpg", "famX/daughter.jpg", 1, "fd")]
        neg = matched_negatives(pos, family, seed=1)
        self.assertEqual(sum(p[3] == "fd" for p in neg), 0)


class TestColourMatchedNegatives(unittest.TestCase):
    """Cross-family negatives bring back the photo-session cue: relatives share
    lighting and colour. Matching on colour distance limits that cue."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.pos = make_positives(60)
        # Positives are photographed together, so their colours nearly agree.
        self.colour = {}
        for a, b, _l, _r in self.pos:
            base = rng.normal(0, 3, 2)
            self.colour[a] = base + rng.normal(0, 1.0, 2)
            self.colour[b] = base + rng.normal(0, 1.0, 2)
        self.distance = lambda x, y: float(
            np.linalg.norm(self.colour[x] - self.colour[y]))

    def colour_auc(self, negatives):
        side = self.pos + negatives
        return roc_auc_score([p[2] for p in side],
                             [-self.distance(p[0], p[1]) for p in side])

    def test_colour_no_longer_separates_the_classes(self):
        plain = matched_negatives(self.pos, family, seed=2)
        matched = colour_matched_negatives(self.pos, family, self.distance, seed=2)
        self.assertGreater(self.colour_auc(plain), 0.75)
        self.assertLess(self.colour_auc(matched), self.colour_auc(plain) - 0.25)

    def test_keeps_the_relation_and_family_guarantees(self):
        negatives = colour_matched_negatives(self.pos, family, self.distance, seed=2)
        self.assertEqual(len(negatives), len(self.pos))
        expected = {"fs": ("father.jpg", "son.jpg"), "md": ("mother.jpg", "daughter.jpg")}
        for a, b, label, relation in negatives:
            self.assertEqual(label, 0)
            self.assertNotEqual(family(a), family(b))
            self.assertEqual((role(a), role(b)), expected[relation])


class TestColourSignature(unittest.TestCase):
    def test_same_chroma_at_different_brightness_is_close(self):
        import os
        import tempfile

        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for name, rgb in (("dark", (60, 60, 60)), ("light", (200, 200, 200)),
                              ("red", (200, 40, 40))):
                paths[name] = os.path.join(folder, name + ".png")
                Image.new("RGB", (8, 8), rgb).save(paths[name])
            dark = colour_signature(paths["dark"])
            light = colour_signature(paths["light"])
            red = colour_signature(paths["red"])
        self.assertLess(np.linalg.norm(dark - light), np.linalg.norm(dark - red))


class TestSplitFitVal(unittest.TestCase):
    def test_families_are_disjoint(self):
        fit, val = split_fit_val(make_positives(), family, seed=3)
        self.assertFalse({family(p[0]) for p in fit} & {family(p[0]) for p in val})

    def test_every_positive_is_kept(self):
        pos = make_positives()
        fit, val = split_fit_val(pos, family, seed=3)
        self.assertEqual(len(fit) + len(val), len(pos))

    def test_rejects_negative_input(self):
        with self.assertRaises(ValueError):
            split_fit_val([("a/x", "b/y", 0, "fs")] * 4, family, seed=3)


class TestBuildSide(unittest.TestCase):
    def test_validation_side_is_class_balanced(self):
        # The old pooled slice was 84% positive because cross-family negatives
        # were discarded when only one endpoint landed in validation.
        fit_pos, val_pos = split_fit_val(make_positives(40), family, seed=3)
        val = build_side(val_pos, family, seed=9)
        labels = [p[2] for p in val]
        self.assertEqual(labels.count(1), labels.count(0))

    def test_positives_without_any_valid_negative_are_excluded(self):
        # Otherwise that relation would be 100% positive: a label leak.
        pos = make_positives(6) + [("famX/father.jpg", "famX/daughter.jpg", 1, "fd")]
        side = build_side(pos, family, seed=9)
        self.assertEqual(sum(p[3] == "fd" for p in side), 0)
        labels = [p[2] for p in side]
        self.assertEqual(labels.count(1), labels.count(0))

    def test_no_endpoint_leaves_the_side(self):
        fit_pos, val_pos = split_fit_val(make_positives(40), family, seed=3)
        allowed = {family(p[0]) for p in val_pos}
        for a, b, _l, _r in build_side(val_pos, family, seed=9):
            self.assertIn(family(a), allowed)
            self.assertIn(family(b), allowed)


class TestDropDuplicateGroups(unittest.TestCase):
    def setUp(self):
        self.pairs = [("fa/f", "fa/c", 1, "fs"), ("fb/f", "fb/c", 1, "fs"),
                      ("fc/f", "fc/c", 1, "fs")]
        base = {"f": np.array([1.0, 0, 0]), "c": np.array([0, 1.0, 0])}
        self.emb = {"fa/f": base["f"], "fa/c": base["c"],
                    "fb/f": base["f"].copy(), "fb/c": base["c"].copy(),
                    "fc/f": np.array([0, 0, 1.0]), "fc/c": np.array([1.0, 1.0, 0])}

    def test_family_stored_twice_is_removed_once(self):
        kept, dropped = drop_duplicate_groups(
            self.pairs, family, lambda p: self.emb[p])
        self.assertEqual(dropped, ["fb"])
        self.assertEqual(sorted({family(p[0]) for p in kept}), ["fa", "fc"])

    def test_single_shared_face_does_not_drop_a_family(self):
        self.emb["fb/c"] = np.array([0, 0.6, 0.8])
        kept, dropped = drop_duplicate_groups(
            self.pairs, family, lambda p: self.emb[p])
        self.assertEqual(dropped, [])
        self.assertEqual(len(kept), 3)


class TestFailedRunRule(unittest.TestCase):
    def test_model_below_untrained_cosine_on_validation_is_failed(self):
        self.assertTrue(is_failed_run(val_auc=0.71, val_cosine_auc=0.74))

    def test_model_above_cosine_is_kept(self):
        self.assertFalse(is_failed_run(val_auc=0.80, val_cosine_auc=0.74))

    def test_slack_tolerates_a_near_tie(self):
        self.assertFalse(is_failed_run(0.735, 0.74, slack=0.02))


class TestTInterval(unittest.TestCase):
    def test_uses_t_not_normal_for_five_folds(self):
        mean, half = t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(mean, 3.0)
        # sd 1.5811, se 0.7071, t(0.975, 4) = 2.7764; 1.96 would give 1.386.
        self.assertAlmostEqual(half, 1.9632, places=3)

    def test_single_value_has_no_defined_interval(self):
        mean, half = t_interval([0.7])
        self.assertAlmostEqual(mean, 0.7)
        self.assertTrue(np.isnan(half))


class TestBootstrapAucDelta(unittest.TestCase):
    def test_precomputed_weighted_auc_matches_sklearn_with_ties(self):
        labels = np.array([-1, 1, -1, 1, 1, -1])
        scores = np.array([0.2, 0.4, 0.4, 0.4, 0.9, 0.2])
        weights = np.array([2, 1, 3, 0, 4, 1])
        actual = _weighted_auc_from_plan(
            _weighted_auc_plan(labels, scores), weights)
        expected = roc_auc_score(labels, scores, sample_weight=weights)
        self.assertAlmostEqual(actual, expected)

    def test_optimized_bootstrap_matches_direct_weighted_auc_resampling(self):
        rng = np.random.default_rng(8)
        labels = np.tile([0, 1], 10)
        scores_a = rng.integers(0, 5, len(labels)).astype(float)
        scores_b = rng.integers(0, 5, len(labels)).astype(float)
        groups_a = np.repeat(np.arange(5), 4)
        groups_b = np.roll(groups_a, 2)
        n_boot, seed = 300, 17
        actual = bootstrap_auc_delta(
            labels, scores_a, scores_b, groups_a, groups_b,
            n_boot=n_boot, seed=seed)

        keys = np.unique(np.concatenate((groups_a, groups_b)))
        lookup = {group: index for index, group in enumerate(keys.tolist())}
        index_a = np.array([lookup[group] for group in groups_a])
        index_b = np.array([lookup[group] for group in groups_b])
        same_family = index_a == index_b
        random = np.random.default_rng(seed)
        deltas = []
        for _ in range(n_boot):
            counts = np.bincount(
                random.integers(0, len(keys), len(keys)), minlength=len(keys))
            weights = counts[index_a] * counts[index_b]
            weights[same_family] = counts[index_a[same_family]]
            selected = weights > 0
            if len(np.unique(labels[selected])) < 2:
                continue
            deltas.append(
                roc_auc_score(labels[selected], scores_a[selected],
                              sample_weight=weights[selected])
                - roc_auc_score(labels[selected], scores_b[selected],
                                sample_weight=weights[selected]))
        expected = np.percentile(deltas, [2.5, 97.5])
        self.assertAlmostEqual(actual["ci_low"], expected[0])
        self.assertAlmostEqual(actual["ci_high"], expected[1])

    def setUp(self):
        rng = np.random.default_rng(0)
        self.groups = np.repeat(np.arange(200), 2)
        self.y = np.tile([1, 0], 200)
        self.good = self.y + rng.normal(0, 0.6, 400)
        self.bad = rng.normal(0, 1, 400)

    def test_clear_difference_excludes_zero(self):
        out = bootstrap_auc_delta(self.y, self.good, self.bad, self.groups,
                                  n_boot=300, seed=1)
        self.assertGreater(out["delta"], 0.2)
        self.assertGreater(out["ci_low"], 0.0)
        self.assertNotIn("p", out)

    def test_identical_scores_give_zero_delta(self):
        out = bootstrap_auc_delta(self.y, self.good, self.good, self.groups,
                                  n_boot=100, seed=1)
        self.assertEqual(out["delta"], 0.0)
        self.assertEqual((out["ci_low"], out["ci_high"]), (0.0, 0.0))

    def test_groups_are_resampled_whole(self):
        # With one family holding every signal pair, resampling pairs would
        # give a tight interval. Resampling families must not.
        y = np.array([1, 0] * 6)
        groups = np.repeat(np.arange(4), 3)
        a = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], dtype=float)
        b = 1 - a
        out = bootstrap_auc_delta(y, a, b, groups, n_boot=300, seed=1)
        self.assertLess(out["ci_low"], 0.0)
        self.assertGreater(out["ci_high"], 0.0)

    def test_both_endpoint_families_are_included(self):
        groups_a = np.repeat(np.arange(4), 3)
        groups_b = np.tile(np.array([10, 11, 12]), 4)
        y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
        a = np.arange(12, dtype=float)
        out = bootstrap_auc_delta(y, a, -a, groups_a, groups_b,
                                  n_boot=300, seed=2)
        self.assertEqual(out["n_groups"], 7)
        self.assertIn("two-way", out["bootstrap"])

    def test_endpoint_group_lengths_must_match(self):
        with self.assertRaises(ValueError):
            bootstrap_auc_delta(self.y, self.good, self.bad, self.groups,
                                self.groups[:-1], n_boot=100)


class TestAdjustPvalues(unittest.TestCase):
    def test_holm_on_a_hand_worked_family(self):
        # Sorted 0.01, 0.02, 0.04 -> x3, x2, x1 = 0.03, 0.04, 0.04.
        got = adjust_pvalues([0.04, 0.01, 0.02], "holm")
        np.testing.assert_allclose(got, [0.04, 0.03, 0.04])

    def test_bh_on_a_hand_worked_family(self):
        # 0.01*3/1=0.03, 0.02*3/2=0.03, 0.04*3/3=0.04.
        got = adjust_pvalues([0.04, 0.01, 0.02], "bh")
        np.testing.assert_allclose(got, [0.04, 0.03, 0.03])

    def test_adjusted_values_never_exceed_one(self):
        self.assertEqual(max(adjust_pvalues([0.9, 0.8], "holm")), 1.0)


if __name__ == "__main__":
    unittest.main()
