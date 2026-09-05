#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import torch

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_mlp15_mediation as candidate
import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast as shared_basic
import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast_interaction_select as interaction
import run_bracket_l13h8_mu_delta_mlp15_mediation as runner


class MLP15MediationTests(unittest.TestCase):
    def test_fresh_aligned_triplets_and_frozen_price(self):
        self.assertEqual(len(candidate.ROWS), 24)
        groups = {}
        for row in candidate.ROWS:
            groups.setdefault(row["group_id"], []).append(row)
        self.assertEqual(len(groups), 8)
        self.assertTrue(all({row["delimiter_index"] for row in rows} == {0, 1, 2}
                            for rows in groups.values()))
        self.assertTrue(all(len({len(row["ids"]) for row in rows}) == 1
                            and len({row["open_position"] for row in rows}) == 1
                            for rows in groups.values()))
        prior = {row["text"] for row in shared_basic.ROWS + interaction.ROWS}
        self.assertFalse({row["text"] for row in candidate.ROWS} & prior)
        plan = candidate.compile_plan()
        self.assertEqual(plan["price"], {"model_forwards": 6, "example_evaluations": 144,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["opened_splits"], ["FRESH_BASIC"])
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])
        self.assertEqual(plan["outcome_reads"], [])

    def test_vector_projection_and_cosine(self):
        native = torch.tensor([2.0, -1.0, -1.0])
        removed = torch.zeros(3)
        exact = runner.vector_metrics(native, removed, native)
        half = runner.vector_metrics(native, removed, native / 2)
        orthogonal = runner.vector_metrics(native, removed, torch.tensor([0.0, 1.0, -1.0]))
        self.assertAlmostEqual(exact["projection_recovery"], 1.0, places=6)
        self.assertAlmostEqual(exact["rescue_cosine"], 1.0, places=6)
        self.assertAlmostEqual(half["projection_recovery"], 0.5, places=6)
        self.assertAlmostEqual(half["rescue_cosine"], 1.0, places=6)
        self.assertAlmostEqual(orthogonal["projection_recovery"], 0.0, places=6)

    def test_thin_dispatch_is_fixed_to_mlp15_and_reruns_suffix(self):
        source = inspect.getsource(runner.factor_mlp15_forward)
        self.assertIn("parent.shared.replay_head", source)
        self.assertIn("event.site == authority.MEDIATOR_LAYER", source)
        self.assertIn("write[arange, finals] = restore_mlp15", source)
        self.assertIn("facade.forward_with_dispatch", source)
        self.assertNotIn("MLP14", inspect.getsource(runner))

    def test_score_requires_live_factor_and_aligned_rescue(self):
        records = []
        for family in candidate.FAMILIES:
            for factor in ("mu", "delta"):
                for index in range(6):
                    records.append({"row_id": f"{family}-{index}", "family_id": family,
                                    "factor": factor, "native_centered_correct_closer": 1.0,
                                    "total_effect_norm": 1.0, "projection_recovery": 0.6,
                                    "rescue_cosine": 0.8, "signed_correct_answer_ce_change": 0.4,
                                    "signed_correct_answer_ce_rescue": 0.3})
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_a"])
        self.assertTrue(report["predictions"]["pred_b"])
        records[0]["total_effect_norm"] = 0.0
        records[1]["total_effect_norm"] = 0.0
        records[2]["total_effect_norm"] = 0.0
        records[3]["total_effect_norm"] = 0.0
        report = runner.score(records, 0.0)
        self.assertFalse(report["instrument_live"])


if __name__ == "__main__":
    unittest.main()
