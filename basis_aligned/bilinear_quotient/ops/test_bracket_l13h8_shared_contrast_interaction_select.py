#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast as basic
import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast_interaction_select as candidate
import run_bracket_l13h8_shared_contrast_interaction_select as runner


class InteractionSelectTests(unittest.TestCase):
    def test_fresh_complete_select_triplets_and_price(self):
        self.assertEqual(len(candidate.ROWS), 24)
        groups = {}
        for row in candidate.ROWS: groups.setdefault(row["group_id"], []).append(row)
        self.assertEqual(len(groups), 8)
        self.assertTrue(all({row["delimiter_index"] for row in rows} == {0, 1, 2}
                            for rows in groups.values()))
        self.assertFalse({row["text"] for row in candidate.ROWS} & {row["text"] for row in basic.ROWS})
        plan = candidate.compile_plan()
        self.assertEqual(plan["price"], {"model_forwards": 6, "example_evaluations": 144,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])

    def test_runner_reuses_parent_and_adds_only_both_removed(self):
        source = inspect.getsource(runner)
        self.assertIn("parent.evaluate", source)
        self.assertIn("replacement_terms=torch.zeros_like(terms)", source)
        self.assertNotIn("def replay_head", source)

    def test_exact_mobius_score_distinguishes_additive_and_interacting(self):
        records = []
        for family in candidate.FAMILIES:
            for index in range(3):
                records.append({"family_id": family, "native_type_axis": 2.0,
                                "native_common_axis": 3.0, "contrast_removed_type_axis": 1.0,
                                "contrast_removed_common_axis": 3.0, "shared_removed_type_axis": 2.0,
                                "shared_removed_common_axis": 1.0, "both_removed_type_axis": 1.0,
                                "both_removed_common_axis": 1.0, "natural_swap_type_transfer": 1.0,
                                "semantic_open_term_norm": 1.0})
        report = runner.score(records, 0.0)
        self.assertTrue(report["additive_oblique_held"])
        self.assertTrue(report["predictions"]["pred_a_instrument_live"])
        self.assertTrue(report["predictions"]["pred_b_additive_oblique"])
        self.assertFalse(report["predictions"]["pred_c_nonlinear_interaction"])
        for row in records[:3]:
            row["both_removed_type_axis"] += 1.0
        report = runner.score(records, 0.0)
        self.assertTrue(report["large_interaction_held"])


if __name__ == "__main__": unittest.main()
