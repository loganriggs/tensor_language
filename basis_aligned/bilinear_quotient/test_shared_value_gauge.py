import unittest

import torch

from .shared_value_gauge import (apply_shared_gauge, canonical_row_basis,
                                 canonicalize_shared_value_bus,
                                 generic_parameter_dimension,
                                 value_output_action)


class SharedValueGaugeTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(2708)
        self.values = [torch.randn(3, 7, generator=generator, dtype=torch.float64)
                       for _ in range(4)]
        self.outputs = [torch.randn(7, 3, generator=generator, dtype=torch.float64)
                        for _ in range(4)]
        self.locals = [torch.randn(7, 5, generator=generator, dtype=torch.float64)
                       for _ in range(4)]
        self.shared_input = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        self.mixing = [.0, .2, .6, .9]
        self.gauge = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        self.gauge += 3*torch.eye(3, dtype=torch.float64)

    def test_one_gauge_shared_across_depth_preserves_every_layer(self):
        expected = value_output_action(self.values, self.outputs, self.locals,
                                       self.shared_input, self.mixing)
        values, outputs = apply_shared_gauge(self.values, self.outputs, self.gauge)
        actual = value_output_action(values, outputs, self.locals,
                                     self.shared_input, self.mixing)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)

    def test_independent_layer_gauge_breaks_shared_value_term(self):
        expected = value_output_action(self.values, self.outputs, self.locals,
                                       self.shared_input, self.mixing)
        values = list(self.values); outputs = list(self.outputs)
        # Transform only layer 2's local V/O pair; its untransformed shared V0 term
        # now meets a changed output basis and must differ when alpha is nonzero.
        values[2] = self.gauge @ values[2]
        outputs[2] = outputs[2] @ torch.linalg.inv(self.gauge)
        actual = value_output_action(values, outputs, self.locals,
                                     self.shared_input, self.mixing)
        self.assertGreater(float((expected[2]-actual[2]).abs().max()), 1e-3)

    def test_canonical_row_basis_depends_only_on_rowspace(self):
        first = canonical_row_basis(self.values[0])
        second = canonical_row_basis(self.gauge @ self.values[0])
        torch.testing.assert_close(first, second, atol=1e-9, rtol=1e-9)

    def test_full_bus_canonicalization_is_gauge_invariant(self):
        first_values, first_outputs = canonicalize_shared_value_bus(
            self.values, self.outputs)
        changed_values, changed_outputs = apply_shared_gauge(
            self.values, self.outputs, self.gauge)
        second_values, second_outputs = canonicalize_shared_value_bus(
            changed_values, changed_outputs)
        for first, second in zip(first_values+first_outputs,
                                 second_values+second_outputs):
            torch.testing.assert_close(first, second, atol=2e-8, rtol=2e-8)

    def test_bilin18_dimension_counts_only_one_gl_per_head(self):
        dimensions = generic_parameter_dimension(18, 1152, 128, 9)
        self.assertEqual(dimensions["raw_parameters"], 47_775_744)
        self.assertEqual(dimensions["continuous_gauge_dimension"], 147_456)
        self.assertEqual(dimensions["quotient_dimension"], 47_628_288)
        self.assertEqual(dimensions["incorrect_independent_layer_quotient_dimension"],
                         45_121_536)

    def test_rank_deficiency_and_invalid_dimensions_fail_closed(self):
        with self.assertRaises(ValueError):
            canonical_row_basis(torch.ones(3, 7, dtype=torch.float64))
        with self.assertRaises(ValueError):
            generic_parameter_dimension(18, 64, 128, 9)


if __name__ == "__main__":
    unittest.main()
