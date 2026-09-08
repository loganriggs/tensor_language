# CPU-only unit tests for factorial_tensor_state.
import unittest

import numpy as np

import factorial_tensor_state as state


class FactorialTensorStateTest(unittest.TestCase):
    def test_suffix_alignment_uses_only_final_valid_positions(self):
        tensor = np.arange(3 * 4 * 5 * 2).reshape(3, 4, 5, 2)
        mask = np.zeros((3, 4, 5), dtype=bool)
        mask[:, :, :4] = True
        mask[0, 0, 0] = False
        aligned, common = state.suffix_align(tensor, mask)
        self.assertEqual(common, 3)
        np.testing.assert_array_equal(aligned[0, 0], tensor[0, 0, 1:4])
        np.testing.assert_array_equal(aligned[2, 3], tensor[2, 3, 1:4])

    def test_cell_program_recovers_cell_constant_product(self):
        rows, positions, factors = 16, 2, 3
        H = np.empty((3, rows, positions, factors), dtype=np.float64)
        R = np.empty_like(H)
        for panel in range(3):
            for row in range(rows):
                direction = row % 2
                H[panel, row] = 1.0 + panel + 0.1 * direction
                R[panel, row] = np.asarray([[1, 2, 3], [3, 2, 1]]) * (1.0 + direction)
        folds = state.balanced_mod4_folds(rows)
        for fit, test in folds.values():
            program = state.contract_state_program(
                state.fit_factorial_states(H, fit), state.fit_factorial_states(R, fit), "cell")
            for panel in range(3):
                for row in test:
                    reference = np.sum(H[panel, row] * R[panel, row], axis=0)
                    cosine, recovery = state.cosine_and_recovery(program[(panel, row % 2)], reference)
                    self.assertAlmostEqual(cosine, 1.0)
                    self.assertAlmostEqual(recovery, 1.0)

    def test_fold_balance_and_bad_shapes_fail_closed(self):
        for fit, test in state.balanced_mod4_folds().values():
            self.assertEqual([np.sum(fit % 2 == d) for d in (0, 1)], [4, 4])
            self.assertEqual([np.sum(test % 2 == d) for d in (0, 1)], [4, 4])
        with self.assertRaises(ValueError): state.balanced_mod4_folds(15)
        with self.assertRaises(ValueError): state.suffix_align(np.zeros((3, 2, 4)), np.zeros((3, 2)))

    def test_gain_models_recover_bilinear_scalar_on_heldout_points(self):
        writer = np.linspace(0.5, 1.5, 12)
        reader = np.linspace(1.4, 0.6, 12)
        target = 0.2 + 1.7 * writer * reader
        fitted = state.fit_gain_models(writer[:8], reader[:8], target[:8])
        predicted = state.predict_gain_models(fitted, writer[8:], reader[8:])
        product = state.scalar_prediction_metrics(predicted["product"], target[8:])
        joint = state.scalar_prediction_metrics(predicted["joint"], target[8:])
        self.assertAlmostEqual(product["r2"], 1.0, places=10)
        self.assertAlmostEqual(joint["r2"], 1.0, places=10)
        self.assertGreater(product["r2"], state.scalar_prediction_metrics(
            predicted["writer"], target[8:])["r2"])

    def test_projection_coefficient_and_gain_shapes_fail_closed(self):
        self.assertAlmostEqual(state.projection_coefficient(np.array([2., 4.]), np.array([1., 2.])), 2.0)
        with self.assertRaises(ValueError): state.projection_coefficient(np.zeros(2), np.zeros(2))
        with self.assertRaises(ValueError): state.gain_design("missing", np.ones(2), np.ones(2))

    def test_hr_balancing_preserves_products_and_spectra_under_reciprocal_gauge(self):
        rng = np.random.default_rng(41)
        writer = rng.normal(size=(3, 4, 2, 5))
        reader = rng.normal(size=writer.shape)
        mask = np.ones(writer.shape[:-1], dtype=bool)
        mask[0, 0, 0] = False
        balanced_h, balanced_r, _scale = state.balance_hr_gauge(writer, reader, mask)
        gauge = np.asarray([-3.0, -0.4, 0.2, 2.0, 7.0])
        changed_h, changed_r, _changed_scale = state.balance_hr_gauge(
            writer * gauge, reader / gauge, mask)
        np.testing.assert_allclose(balanced_h * balanced_r, writer * reader, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(changed_h * changed_r, writer * reader, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            state.masked_factor_spectrum(balanced_h, mask),
            state.masked_factor_spectrum(changed_h, mask), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            state.masked_factor_spectrum(balanced_r, mask),
            state.masked_factor_spectrum(changed_r, mask), rtol=1e-12, atol=1e-12)

    def test_hr_balancing_rejects_unoccupied_factors(self):
        writer = np.ones((2, 3, 4))
        reader = np.ones_like(writer)
        writer[..., 2] = 0
        with self.assertRaises(ValueError):
            state.balance_hr_gauge(writer, reader, np.ones((2, 3), dtype=bool))


if __name__ == "__main__": unittest.main()
