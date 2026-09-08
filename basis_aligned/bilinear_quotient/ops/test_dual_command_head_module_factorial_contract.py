#!/usr/bin/env python3
import unittest

import dual_command_head_module_factorial_contract as contract


class DualCommandHeadModuleContractTests(unittest.TestCase):
    def test_pairing_is_an_involution_and_role_specific(self):
        for cell in ("00", "10", "01", "11"):
            for role in ("temporal", "iswas"):
                paired = contract.paired_cell(cell, role)
                self.assertEqual(contract.paired_cell(paired, role), cell)
                untouched = 1 if role == "temporal" else 0
                self.assertEqual(paired[untouched], cell[untouched])

    def test_exact_effect_metrics(self):
        metrics = contract.effect_metrics([1, -2], [1, -2], [0, 0])
        self.assertAlmostEqual(metrics["signed_recovery"], 1)
        self.assertAlmostEqual(metrics["cosine"], 1)
        self.assertAlmostEqual(metrics["relative_residual"], 0)
        self.assertAlmostEqual(metrics["non_target_to_target_gold_norm"], 0)

    def test_aggregate_retains_template_and_phase(self):
        records = [{"role": "temporal", "phase": "FIT", "template_id": template,
                    "arm": "L09H01", "target_effect": 1, "target_gold": 1,
                    "non_target_effect": 0} for template in ("a", "b")]
        reports = contract.aggregate(records)
        self.assertEqual(len(reports), 3)
        self.assertEqual(next(row for row in reports if row["template_id"] == "ALL")["count"], 2)

    def test_union_additivity_and_guards(self):
        self.assertAlmostEqual(contract.union_additivity([3, 3], ([1, 1], [2, 2])), 0)
        self.assertEqual(contract.union_additivity([0, 0], ([1, 1],)), float("inf"))
        with self.assertRaises(ValueError): contract.effect_metrics([1], [0], [0])
        with self.assertRaises(ValueError): contract.paired_cell("10", "other")


if __name__ == "__main__": unittest.main()
