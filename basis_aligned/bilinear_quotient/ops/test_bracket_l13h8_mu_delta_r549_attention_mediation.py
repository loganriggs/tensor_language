#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_mlp15_mediation as rows_authority
import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_r549_attention_mediation as candidate
import run_bracket_l13h8_mu_delta_r549_attention_mediation as runner


class R549AttentionMediationTests(unittest.TestCase):
    def test_reuses_frozen_rows_and_prices_exact_ten_forwards(self):
        self.assertIs(candidate.ROWS, rows_authority.ROWS)
        self.assertEqual(candidate.ROWS_SHA256,
                         "61f8db59d41863f2cc48b141f7c14f73a6774c63bbbca3b41ae0d873dbd18ba7")
        plan = candidate.compile_plan()
        self.assertEqual(plan["fixed_heads"], ["L14H1", "L15H3", "L16H1"])
        self.assertEqual(len(plan["conditions"]), 10)
        self.assertEqual(plan["price"], {"model_forwards": 10, "example_evaluations": 240,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])
        self.assertEqual(plan["outcome_reads"], [])

    def test_dispatch_restores_only_one_exact_post_ov_head_contribution(self):
        source = inspect.getsource(runner.attention_factor_forward)
        self.assertIn("base.parent.shared.replay_head", source)
        self.assertIn("restore_contribution - contribution", source)
        self.assertIn("write[arange, finals]", source)
        self.assertIn("facade.forward_with_dispatch", source)
        self.assertNotIn("mlp15_final", source)
        self.assertNotIn("rank", source.lower())

    def test_score_distinguishes_fixed_head_mediation_from_bypass(self):
        records = []
        for family in candidate.FAMILIES:
            for factor in ("mu", "delta"):
                for head in ("L14H1", "L15H3", "L16H1"):
                    for index in range(6):
                        good = head == "L15H3"
                        records.append({
                            "row_id": f"{family}-{index}", "family_id": family,
                            "factor": factor, "head": head,
                            "native_centered_correct_closer": 1.0,
                            "total_effect_norm": 1.0,
                            "projection_recovery": 0.4 if good else 0.0,
                            "rescue_cosine": 0.7 if good else 0.0,
                            "signed_correct_answer_ce_change": 0.3,
                            "signed_correct_answer_ce_rescue": 0.2 if good else 0.0,
                        })
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_a"])
        self.assertTrue(report["predictions"]["pred_b"])
        self.assertEqual(report["qualifying_heads_by_factor_on_target_constructions"],
                         {"mu": ["L15H3"], "delta": ["L15H3"]})
        for row in records:
            if row["factor"] == "delta":
                row["projection_recovery"] = 0.0
                row["rescue_cosine"] = 0.0
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_c"])
        self.assertFalse(report["predictions"]["pred_b"])


if __name__ == "__main__":
    unittest.main()
