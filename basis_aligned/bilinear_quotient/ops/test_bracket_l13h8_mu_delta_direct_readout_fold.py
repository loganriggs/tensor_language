#!/usr/bin/env python3

from __future__ import annotations

import inspect
import types
import unittest

import torch
import torch.nn.functional as F

import circuit_fast_screen_candidate_bracket_l13h8_mu_delta_direct_readout_fold as candidate
import run_bracket_l13h8_mu_delta_direct_readout_fold as runner


class DirectReadoutFoldTests(unittest.TestCase):
    def test_frozen_rows_three_forward_price_and_splits(self):
        plan = candidate.compile_plan()
        self.assertEqual(candidate.ROWS_SHA256,
                         "61f8db59d41863f2cc48b141f7c14f73a6774c63bbbca3b41ae0d873dbd18ba7")
        self.assertEqual(len(plan["conditions"]), 3)
        self.assertEqual(plan["price"], {"model_forwards": 3, "example_evaluations": 72,
                                         "backwards": 0, "parameter_updates": 0})
        self.assertEqual(plan["outcome_reads"], [])
        self.assertEqual(plan["closed_splits"], ["TEST", "OOD"])

    def test_two_scale_least_squares_recovers_exact_synthetic_identity(self):
        generator = torch.Generator().manual_seed(17)
        z_native = torch.randn(12, generator=generator)
        z_removed = torch.randn(12, generator=generator)
        factor = 2.3 * z_native - 1.7 * z_removed
        r_native, r_removed, max_error, relative = runner.solve_rms_scales(
            z_native, z_removed, factor, torch)
        self.assertAlmostEqual(float(r_native), 2.3, places=5)
        self.assertAlmostEqual(float(r_removed), 1.7, places=5)
        self.assertLess(max_error, 2e-6)
        self.assertLess(relative, 1e-6)

    def test_weight_fold_normalization_and_softcap_reconstruct_exact_difference(self):
        generator = torch.Generator().manual_seed(23)
        z_native = torch.randn(7, generator=generator)
        z_removed = torch.randn(7, generator=generator)
        r_native, r_removed = torch.tensor(1.4), torch.tensor(0.9)
        factor = r_native * z_native - r_removed * z_removed
        weight = torch.randn(11, 7, generator=generator)
        components = runner.readout_components(
            weight, None, z_native, z_removed, factor,
            r_native, r_removed, torch, F)
        torch.testing.assert_close(
            components["direct"] + components["normalization"],
            components["raw_difference"], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            components["direct"] + components["normalization"] + components["softcap"],
            components["final_difference"], atol=2e-6, rtol=2e-6)

    def test_hook_is_temporary_and_runner_uses_exactly_three_model_forwards(self):
        layer = torch.nn.Linear(4, 3, bias=False)
        model = types.SimpleNamespace(lm_head=layer)
        output, captured = runner.capture_lm_head_input(
            model, lambda: layer(torch.ones(2, 4)))
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertEqual(tuple(captured.shape), (2, 4))
        self.assertEqual(len(layer._forward_pre_hooks), 0)
        source = inspect.getsource(runner.evaluate)
        self.assertEqual(source.count("capture_lm_head_input("), 2)
        self.assertIn('for factor in ("mu", "delta")', source)
        self.assertIn("install_bank=native_bank", source)


if __name__ == "__main__":
    unittest.main()
