#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_downstream_module_mediation as candidate
import run_bracket_l13h8_mu_delta_downstream_module_mediation as runner


class DownstreamModuleMediationTests(unittest.TestCase):
    def test_frozen_rows_modules_price_and_splits(self):
        plan = candidate.compile_plan()
        self.assertEqual(candidate.ROWS_SHA256,
                         "61f8db59d41863f2cc48b141f7c14f73a6774c63bbbca3b41ae0d873dbd18ba7")
        self.assertEqual(candidate.MODULES,
                         ("mlp13", "attention14", "mlp14", "attention15",
                          "attention16", "mlp16", "attention17", "mlp17"))
        self.assertNotIn("mlp15", candidate.MODULES)
        self.assertEqual(len(plan["conditions"]), 20)
        self.assertEqual(plan["price"], {"model_forwards": 20, "example_evaluations": 480,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["outcome_reads"], [])
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])

    def test_local_dispatch_restores_complete_module_final_write(self):
        source = inspect.getsource(runner.module_factor_forward)
        self.assertIn("base.parent.shared.replay_head", source)
        self.assertIn("write[arange, finals] = restore_write", source)
        self.assertIn("event.block.attn", source)
        self.assertIn("event.block.mlp", source)
        self.assertNotIn("replay_attention_with_head", source)

    def test_score_selects_module_only_when_both_target_families_pass(self):
        records = []
        for family in candidate.FAMILIES:
            for factor in ("mu", "delta"):
                for module in candidate.MODULES:
                    for index in range(6):
                        good = module == "attention16"
                        records.append({
                            "row_id": f"{family}-{index}", "family_id": family,
                            "factor": factor, "module": module,
                            "native_centered_correct_closer": 1.0,
                            "total_effect_norm": 1.0,
                            "projection_recovery": 0.3 if good else 0.0,
                            "rescue_cosine": 0.6 if good else 0.0,
                            "signed_correct_answer_ce_change": 0.4,
                            "signed_correct_answer_ce_rescue": 0.2 if good else 0.0,
                        })
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_b"])
        self.assertEqual(report["qualifying_modules_by_factor_on_target_constructions"],
                         {"mu": ["attention16"], "delta": ["attention16"]})
        for row in records:
            if (row["factor"] == "delta" and row["family_id"] == "direct_type"
                    and row["module"] == "attention16"):
                row["projection_recovery"] = 0.0
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_c"])


if __name__ == "__main__":
    unittest.main()
