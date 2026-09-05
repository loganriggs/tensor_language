#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import sys
import unittest

import torch

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as candidate
import run_bracket_l13h8_source_region_payload_factorial as runner


class CandidateTests(unittest.TestCase):
    def test_exact_balanced_fresh_authority(self):
        rows = candidate.ROWS
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["row_id"] for row in rows}), 24)
        self.assertEqual({family: sum(row["family_id"] == family for row in rows)
                          for family in candidate.FAMILIES}, {family: 6 for family in candidate.FAMILIES})
        self.assertTrue(all(row["split"] == "BASIC_SCREEN" for row in rows))

    def test_regions_are_disjoint_exhaustive_and_aligned(self):
        for row in candidate.ROWS:
            regions = row["regions"]
            flattened = sum((regions[name] for name in candidate.REGIONS), [])
            self.assertEqual(sorted(flattened), list(range(len(row["base_ids"]))))
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertEqual(row["base_open_position"], row["donor_open_position"])
            self.assertEqual(regions["OPEN"], [row["base_open_position"]])
            self.assertEqual(len(row["base_ids"]), len(row["donor_ids"]))

    def test_target_and_same_state_control_contracts(self):
        for row in candidate.ROWS:
            changes = row["base_answer_id"] != row["donor_answer_id"]
            self.assertEqual(changes, row["role"] == "target")
            self.assertNotEqual(row["base_ids"], row["donor_ids"])

    def test_all_eight_corners_and_price_are_frozen(self):
        self.assertEqual(len(candidate.CORNERS), 8)
        self.assertEqual(set(candidate.CORNERS), {
            (), ("PREFIX",), ("OPEN",), ("POST",), ("PREFIX", "OPEN"),
            ("PREFIX", "POST"), ("OPEN", "POST"), ("PREFIX", "OPEN", "POST"),
        })
        plan = candidate.compile_plan()
        self.assertEqual(plan["price"], {"model_forwards": 22, "example_evaluations": 528,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["outcome_reads"], [])
        self.assertEqual(plan["closed_splits"], ["SELECT", "FINAL_TEST", "OOD"])
        self.assertEqual(plan["bars"]["native_answer_positive_fraction_min"], 0.75)
        self.assertEqual(plan["bars"]["open_post_target_median_recovery_min"], 0.50)

    def test_native_capability_is_scored_per_family_and_direction(self):
        capability = []
        for family in candidate.FAMILIES:
            for direction in runner.DIRECTIONS:
                capability.extend({"family_id": family, "direction": direction,
                                   "answer_margin": 1.0} for _ in range(6))
        raw = []
        for row in candidate.ROWS:
            for direction in runner.DIRECTIONS:
                for condition, effect, normalized in (
                    ("complete_head", 1.0, 1.0),
                    ("payload_OPEN+POST", 0.8 if row["role"] == "target" else 0.0,
                     0.8 if row["role"] == "target" else 0.0),
                    ("payload_PREFIX", 0.0, 0.0),
                    ("payload_PREFIX+OPEN+POST", 0.8 if row["role"] == "target" else 0.0,
                     0.8 if row["role"] == "target" else 0.0),
                ):
                    raw.append({"role": row["role"], "condition": condition,
                                "effect": effect, "normalized_effect": normalized})
        scored = runner.score_basic_screen(raw, 0.0, capability)
        self.assertTrue(scored["checks"]["native_capability"])
        self.assertEqual(scored["predictions"], {
            "pred_a_instrument_live": True,
            "pred_b_open_post_localized": True,
            "pred_c_broad_or_distributed": False,
        })
        capability[0]["answer_margin"] = -1.0
        capability[1]["answer_margin"] = -1.0
        self.assertFalse(runner.score_basic_screen(raw, 0.0, capability)["checks"]["native_capability"])

    def test_payload_swap_retains_recipient_scores(self):
        p = torch.tensor([[2.0, 3.0, 5.0]])
        recipient = torch.tensor([[[1.0], [10.0], [100.0]]])
        donor = torch.tensor([[[7.0], [11.0], [13.0]]])
        selected = torch.tensor([[False, True, False]])
        got = runner.payload_hybrid(p, recipient, donor, selected, torch)
        self.assertTrue(torch.equal(got, torch.tensor([[535.0]])))

    def test_mobius_transform_recovers_pair_interaction(self):
        raw = []
        for corner in candidate.CORNERS:
            subset = set(corner)
            value = sum({"PREFIX": 1.0, "OPEN": 2.0, "POST": 4.0}[item] for item in subset)
            if {"OPEN", "POST"} <= subset:
                value += 8.0
            raw.append({"row_id": "r", "group_id": "g", "family_id": "direct_type",
                        "direction": "base_to_donor", "condition": "payload_" + runner.corner_name(corner),
                        "normalized_effect": value})
        coefficients = runner.mobius_interactions(raw)[0]["coefficients"]
        self.assertEqual(coefficients["OPEN+POST"], 8.0)
        self.assertEqual(coefficients["PREFIX+OPEN+POST"], 0.0)

    def test_import_is_cpu_only_and_does_not_change_execution_mode(self):
        old = os.environ.pop("BQLIB_NO_MODEL", None)
        try:
            importlib.reload(runner)
            self.assertNotIn("BQLIB_NO_MODEL", os.environ)
        finally:
            if old is not None:
                os.environ["BQLIB_NO_MODEL"] = old


if __name__ == "__main__":
    unittest.main()
