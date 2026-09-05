#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import torch

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_residual_write_bank_factorial as candidate
import run_bracket_l13h8_mu_delta_residual_write_bank_factorial as runner


class ResidualWriteBankFactorialTests(unittest.TestCase):
    def test_frozen_rows_bank_price_and_splits(self):
        plan = candidate.compile_plan()
        self.assertEqual(candidate.ROWS_SHA256,
                         "61f8db59d41863f2cc48b141f7c14f73a6774c63bbbca3b41ae0d873dbd18ba7")
        self.assertEqual(len(candidate.WRITE_BANK), 9)
        self.assertEqual(len(plan["conditions"]), 8)
        self.assertEqual(plan["price"], {"model_forwards": 8, "example_evaluations": 192,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["outcome_reads"], [])
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])

    def test_factorial_identities_and_projection(self):
        nn = torch.tensor([2.0, -1.0, -1.0])
        rn = torch.tensor([1.4, -0.7, -0.7])
        nr = torch.tensor([1.6, -0.8, -0.8])
        rr = torch.tensor([1.0, -0.5, -0.5])
        total, residual, write = nn - rr, nn - rn, nn - nr
        interaction = nn - rn - nr + rr
        torch.testing.assert_close(residual + write - interaction, total)
        self.assertAlmostEqual(runner.projection(residual, total), 0.6, places=6)
        l_nn, l_rn, l_nr, l_rr = 1.0, 1.3, 1.2, 1.8
        total_l, residual_l, write_l = l_rr - l_nn, l_rn - l_nn, l_nr - l_nn
        interaction_l = l_rr - l_rn - l_nr + l_nn
        self.assertAlmostEqual(total_l, residual_l + write_l + interaction_l)

    def test_dispatch_captures_and_installs_whole_bank_without_parent_edit(self):
        source = inspect.getsource(runner.bank_factor_forward)
        self.assertIn("base.parent.shared.replay_head", source)
        self.assertIn("write[arange, finals] = install_bank[name]", source)
        self.assertIn("capture_bank", source)
        self.assertNotIn("replay_attention_with_head", source)

    def test_score_classifies_residual_and_material_downstream(self):
        records = []
        for family in candidate.FAMILIES:
            for factor in ("mu", "delta"):
                for index in range(6):
                    records.append({
                        "row_id": f"{family}-{index}", "family_id": family, "factor": factor,
                        "native_centered_correct_closer": 1.0, "total_effect_norm": 1.0,
                        "residual_path_projection": 0.8, "write_bank_projection": 0.3,
                        "interaction_projection": 0.1,
                        "vector_identity_max_absolute_error": 0.0,
                        "signed_correct_answer_ce_total_damage": 0.5,
                        "signed_correct_answer_ce_residual_path_damage": 0.3,
                        "signed_correct_answer_ce_write_bank_damage": 0.1,
                        "signed_correct_answer_ce_loss_interaction": 0.1,
                        "ce_identity_absolute_error": 0.0,
                    })
        report = runner.score(records, 0.0)
        self.assertTrue(report["predictions"]["pred_a"])
        self.assertTrue(report["predictions"]["pred_b"])
        self.assertTrue(report["predictions"]["pred_c"])


if __name__ == "__main__":
    unittest.main()
