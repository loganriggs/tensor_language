#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_open_post_confirm as prior
import circuit_fast_screen_candidate_bracket_l13h8_open_zero_removal as candidate
import run_bracket_l13h8_semantic_open_zero_removal as runner


class OpenRemovalTests(unittest.TestCase):
    def test_fresh_heldout_authority_and_price(self):
        self.assertEqual(len(candidate.ROWS), 24)
        self.assertTrue(all(row["split"] == "FRESH_HELDOUT_REMOVAL" for row in candidate.ROWS))
        old = {row[key] for row in prior.ROWS for key in ("base_text", "donor_text")}
        new = {row[key] for row in candidate.ROWS for key in ("base_text", "donor_text")}
        self.assertFalse(old & new)
        plan = candidate.compile_plan()
        self.assertEqual(plan["price"], {"model_forwards": 8, "example_evaluations": 192,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["closed_splits"], ["FINAL_TEST", "OOD"])
        self.assertEqual(plan["outcome_reads"], [])

    def test_runner_reuses_exact_factor_executor(self):
        source = inspect.getsource(runner)
        self.assertIn("shared.factor_forward", source)
        self.assertNotIn("def replay_head", source)

    def test_selective_necessity_score_holds_synthetic_case(self):
        records, native = [], []
        for family in candidate.FAMILIES:
            role = "target" if family in candidate.TARGET_FAMILIES else "control"
            for direction in ("base_to_donor", "donor_to_base"):
                native.append({"family_id": family, "direction": direction, "answer_margin": 2.0})
                for i in range(2):
                    damage = 0.8 if role == "target" else 0.01
                    records.append({"family_id": family, "direction": direction, "role": role,
                                    "complete_head_margin_damage": 1.0,
                                    "semantic_open_margin_damage": damage,
                                    "normalized_damage": damage, "answer_preserved": True})
        report = runner.score(records, 0.0, native, [1.0] * len(records))
        self.assertTrue(report["instrument_live"])
        self.assertTrue(report["selective_necessity_held"])


if __name__ == "__main__": unittest.main()
