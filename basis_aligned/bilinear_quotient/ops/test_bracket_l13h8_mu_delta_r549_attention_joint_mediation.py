#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_r549_attention_joint_mediation as candidate
import run_bracket_l13h8_mu_delta_r549_attention_joint_mediation as runner


class JointAttentionMediationTests(unittest.TestCase):
    def test_frozen_rows_six_forward_plan_and_closed_splits(self):
        plan = candidate.compile_plan()
        self.assertEqual(candidate.ROWS_SHA256,
                         "61f8db59d41863f2cc48b141f7c14f73a6774c63bbbca3b41ae0d873dbd18ba7")
        self.assertEqual(len(plan["conditions"]), 6)
        self.assertEqual(plan["price"], {"model_forwards": 6, "example_evaluations": 144,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["outcome_reads"], [])
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])

    def test_plural_restoration_is_successor_local_and_exact(self):
        source = inspect.getsource(runner.joint_attention_factor_forward)
        self.assertIn("parent.replay_attention_with_head", source)
        self.assertIn("restore_contributions[site] - contribution", source)
        self.assertIn("write[arange, finals]", source)
        self.assertNotIn("subset", source.lower())

    def test_interaction_append_and_target_gate(self):
        records, frozen = [], {}
        for family in candidate.FAMILIES:
            for factor in ("mu", "delta"):
                for index in range(6):
                    row_id = f"{family}-{index}"
                    records.append({
                        "row_id": row_id, "family_id": family, "factor": factor,
                        "native_centered_correct_closer": 1.0, "total_effect_norm": 1.0,
                        "projection_recovery": 0.3, "rescue_cosine": 0.6,
                        "signed_correct_answer_ce_change": 0.4,
                        "signed_correct_answer_ce_rescue": 0.2,
                    })
                    frozen[(row_id, factor)] = {"projection_recovery": 0.03, "ce_rescue": 0.02}
        runner.append_frozen_individual_comparison(records, frozen)
        self.assertAlmostEqual(records[0]["joint_minus_sum_individual_projection_recovery"], 0.27)
        self.assertAlmostEqual(records[0]["joint_minus_sum_individual_ce_rescue"], 0.18)
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_b"])
        for row in records:
            if row["family_id"] == "direct_type" and row["factor"] == "delta":
                row["projection_recovery"] = 0.0
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_c"])


if __name__ == "__main__":
    unittest.main()
