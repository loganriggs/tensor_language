#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_open_post_confirm as candidate
import circuit_fast_screen_candidate_bracket_l13h8_source_regions as basic
import circuit_fast_screen_candidate_bracket_l13h8_source_regions_select as select
import run_bracket_l13h8_open_post_family_confirm as runner


class OpenPostConfirmTests(unittest.TestCase):
    def test_fresh_authority_and_narrow_plan(self):
        self.assertEqual(len(candidate.ROWS), 24)
        new = {row[key] for row in candidate.ROWS for key in ("base_text", "donor_text")}
        prior = {row[key] for rows in (basic.ROWS, select.ROWS) for row in rows
                 for key in ("base_text", "donor_text")}
        self.assertFalse(new & prior)
        plan = candidate.compile_plan()
        self.assertEqual(candidate.CORNERS, (("OPEN",), ("POST",), ("OPEN", "POST")))
        self.assertEqual(plan["price"], {"model_forwards": 12, "example_evaluations": 288,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["closed_splits"], ["FINAL_TEST", "OOD"])
        self.assertEqual(plan["outcome_reads"], [])

    def test_thin_facade_reuses_exact_executor(self):
        source = inspect.getsource(runner)
        self.assertIn("shared.evaluate", source)
        self.assertNotIn("def replay_head", source)
        self.assertNotIn("def factor_forward", source)

    def test_frozen_opener_prediction_scores_as_held(self):
        raw, native = [], []
        target_values = {
            "direct_type": (1.0, 0.01, 1.011),
            "completed_then_reopened": (1.6, 0.02, 1.621),
        }
        for family in candidate.FAMILIES:
            role = "target" if family in candidate.TARGET_FAMILIES else "control"
            for direction in ("base_to_donor", "donor_to_base"):
                native.append({"family_id": family, "direction": direction, "answer_margin": 2.0})
                for row_index in range(2):
                    values = target_values.get(family, (0.01, 0.01, 0.02))
                    for condition, value in zip(("payload_OPEN", "payload_POST", "payload_OPEN+POST"), values):
                        raw.append({"row_id": f"{family}-{direction}-{row_index}", "family_id": family,
                                    "direction": direction, "condition": condition, "role": role,
                                    "effect": value, "normalized_effect": value})
                    raw.append({"row_id": f"{family}-{direction}-{row_index}", "family_id": family,
                                "direction": direction, "condition": "complete_head", "role": role,
                                "effect": 1.0, "normalized_effect": 1.0})
        report = runner.score(raw, 0.0, native)
        self.assertTrue(report["instrument_live"])
        self.assertTrue(report["opener_payload_hypothesis_held"])
        self.assertEqual(report["predictions"], {
            "pred_a_instrument_live": True,
            "pred_b_opener_payload_held": True,
            "pred_c_post_or_synergy_material": False,
        })
        self.assertTrue(all(item["passed"] for item in report["family_direction_metrics"].values()))


if __name__ == "__main__":
    unittest.main()
