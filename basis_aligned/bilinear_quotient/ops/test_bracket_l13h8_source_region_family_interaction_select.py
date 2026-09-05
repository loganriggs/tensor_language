#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as basic
import circuit_fast_screen_candidate_bracket_l13h8_source_regions_select as select
import run_bracket_l13h8_source_region_family_interaction_select as runner


class SelectConfirmationTests(unittest.TestCase):
    def test_fresh_balanced_select_authority(self):
        self.assertEqual(len(select.ROWS), 24)
        self.assertTrue(all(row["split"] == "SELECT" for row in select.ROWS))
        self.assertEqual({family: sum(row["family_id"] == family for row in select.ROWS)
                          for family in select.FAMILIES}, {family: 6 for family in select.FAMILIES})
        basic_text = {row[key] for row in basic.ROWS for key in ("base_text", "donor_text")}
        select_text = {row[key] for row in select.ROWS for key in ("base_text", "donor_text")}
        self.assertFalse(basic_text & select_text)

    def test_only_two_confirmatory_payload_conditions(self):
        plan = select.compile_plan()
        self.assertEqual(select.CORNERS, (("PREFIX",), ("OPEN", "POST")))
        self.assertEqual(plan["conditions"], ["native", "native_replay", "complete_head",
                                               "payload_PREFIX", "payload_OPEN+POST"])
        self.assertEqual(plan["price"], {"model_forwards": 10, "example_evaluations": 240,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["opened_splits"], ["SELECT"])
        self.assertEqual(plan["closed_splits"], ["FINAL_TEST", "OOD"])
        self.assertEqual(plan["outcome_reads"], [])

    def test_facade_reuses_shared_executor(self):
        source = inspect.getsource(runner)
        self.assertIn("shared.evaluate", source)
        self.assertNotIn("def replay_head", source)
        self.assertNotIn("def factor_forward", source)

    def test_family_specific_scoring_holds_synthetic_prediction(self):
        raw, native = [], []
        values = {
            "direct_type": {"payload_PREFIX": 0.0, "payload_OPEN+POST": 1.0},
            "completed_then_reopened": {"payload_PREFIX": -0.8, "payload_OPEN+POST": 1.6},
            "same_state_surface": {"payload_PREFIX": 0.01, "payload_OPEN+POST": 0.01},
            "same_state_punctuation": {"payload_PREFIX": 0.01, "payload_OPEN+POST": 0.01},
        }
        for family in select.FAMILIES:
            role = "target" if family in select.TARGET_FAMILIES else "control"
            for direction in ("base_to_donor", "donor_to_base"):
                native.append({"family_id": family, "direction": direction, "answer_margin": 2.0})
                for condition in ("complete_head", "payload_PREFIX", "payload_OPEN+POST"):
                    normalized = 1.0 if condition == "complete_head" else values[family][condition]
                    raw.append({"family_id": family, "direction": direction, "condition": condition,
                                "role": role, "effect": normalized, "normalized_effect": normalized})
        report = runner.score_select(raw, 0.0, native)
        self.assertTrue(report["instrument_live"])
        self.assertTrue(report["family_interaction_held"])
        self.assertTrue(all(cell["passed"] for cell in report["family_direction_metrics"].values()))


if __name__ == "__main__":
    unittest.main()
