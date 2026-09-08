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


if __name__ == "__main__": unittest.main()
