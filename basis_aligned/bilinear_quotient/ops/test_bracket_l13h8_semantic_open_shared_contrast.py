#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast as candidate
import run_bracket_l13h8_semantic_open_shared_contrast as runner
import run_bracket_l13h8_source_region_payload_factorial as shared


class SharedContrastScreenTests(unittest.TestCase):
    def test_authority_is_complete_aligned_triplets(self):
        self.assertEqual(len(candidate.ROWS), 24)
        groups = {}
        for row in candidate.ROWS: groups.setdefault(row["group_id"], []).append(row)
        self.assertEqual(len(groups), 8)
        for rows in groups.values():
            self.assertEqual({row["delimiter_index"] for row in rows}, {0, 1, 2})
            self.assertEqual(len({len(row["ids"]) for row in rows}), 1)
            self.assertEqual(len({row["open_position"] for row in rows}), 1)

    def test_plan_is_five_forwards_and_keeps_later_splits_closed(self):
        plan = candidate.compile_plan()
        self.assertEqual(plan["price"], {"model_forwards": 5, "example_evaluations": 120,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["opened_splits"], ["FRESH_BASIC"])
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])
        self.assertEqual(plan["outcome_reads"], [])
        self.assertEqual(set(plan["frozen_predictions"]), {"pred_a", "pred_b", "pred_c"})

    def test_exact_executor_exposes_term_replacement_without_copying_replay(self):
        shared_source, runner_source = inspect.getsource(shared.factor_forward), inspect.getsource(runner)
        self.assertIn("replacement_terms", shared_source)
        self.assertIn("shared.factor_forward", runner_source)
        self.assertNotIn("def replay_head", runner_source)

    def test_live_ratio_scoring_can_hold_and_fail(self):
        records = []
        for family in candidate.FAMILIES:
            for bundle in range(2):
                for delimiter in range(3):
                    records.append({"family_id": family, "bundle_id": bundle, "delimiter_index": delimiter,
                                    "native_type_axis": 2.0, "natural_swap_type_transfer": 1.0,
                                    "contrast_removal_type_damage": 1.0,
                                    "contrast_type_to_common_ratio": 4.0,
                                    "contrast_normalized_type_damage": 0.5,
                                    "shared_common_to_type_ratio": 3.0,
                                    "semantic_open_term_norm": 1.0})
        self.assertTrue(runner.score(records, 0.0)["shared_plus_contrast_held"])
        records[0]["contrast_type_to_common_ratio"] = 0.1
        records[1]["contrast_type_to_common_ratio"] = 0.1
        records[2]["contrast_type_to_common_ratio"] = 0.1
        records[3]["contrast_type_to_common_ratio"] = 0.1
        self.assertFalse(runner.score(records, 0.0)["target_families"]["direct_type"]["passed"])


if __name__ == "__main__": unittest.main()
