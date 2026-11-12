import unittest

from scripts.evaluation.audit_rfiw_test_coverage import (
    bootstrap_coverage_difference,
    rate_table,
)


class TestRfiwCoverageAudit(unittest.TestCase):
    def test_rates_keep_label_and_relation_denominators(self):
        records = [
            {"label": 1, "relation": "fd", "covered": True},
            {"label": 1, "relation": "fd", "covered": False},
            {"label": 0, "relation": "bb", "covered": False},
        ]
        labels = rate_table(records, "label")
        relations = rate_table(records, "relation")
        self.assertEqual(labels["1"], {
            "total": 2, "covered": 1, "coverage_rate": 0.5})
        self.assertEqual(relations["bb"]["coverage_rate"], 0.0)

    def test_family_bootstrap_uses_both_pair_endpoints(self):
        records = []
        for family in range(20):
            records.extend((
                {"label": 1, "family_a": family, "family_b": family,
                 "covered": True},
                {"label": 0, "family_a": family, "family_b": family,
                 "covered": False},
            ))
        result = bootstrap_coverage_difference(records, n_boot=200, seed=3)
        self.assertEqual(result["families"], 20)
        self.assertEqual(result["difference_percentage_points"], 100.0)
        self.assertEqual(result["ci_low_percentage_points"], 100.0)
        self.assertEqual(result["ci_high_percentage_points"], 100.0)


if __name__ == "__main__":
    unittest.main()
