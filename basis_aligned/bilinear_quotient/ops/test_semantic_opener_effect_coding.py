#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

import semantic_opener_effect_coding as coding


class EffectCodingTests(unittest.TestCase):
    def setUp(self):
        self.terms = (np.array([4.0, 1.0]), np.array([1.0, 4.0]), np.array([1.0, 1.0]))
        self.shared, self.contrasts = coding.centered_terms(self.terms)

    def test_exact_reconstruction_and_zero_sum_fix_the_gauge(self):
        np.testing.assert_allclose(sum(self.contrasts), np.zeros(2), atol=1e-15)
        for term, contrast in zip(self.terms, self.contrasts):
            np.testing.assert_allclose(self.shared + contrast, term, atol=1e-15)

    def test_contrast_swap_equals_natural_term_swap(self):
        head = np.array([20.0, 30.0])
        for recipient in range(3):
            for donor in range(3):
                got = coding.swap_contrast(head, self.terms[recipient], self.shared,
                                           self.contrasts[donor])
                expected = coding.replace_term(head, self.terms[recipient], self.terms[donor])
                np.testing.assert_allclose(got, expected, atol=1e-15)

    def test_removals_preserve_named_complement(self):
        head = np.array([20.0, 30.0])
        for term, contrast in zip(self.terms, self.contrasts):
            contrast_removed = coding.remove_contrast(head, term, self.shared)
            np.testing.assert_allclose(contrast_removed - (head - term), self.shared)
            shared_removed = coding.remove_shared(head, self.shared)
            np.testing.assert_allclose(shared_removed - (head - term), contrast)

    def test_permutation_equivariance(self):
        permuted = (self.terms[2], self.terms[0], self.terms[1])
        shared, contrasts = coding.centered_terms(permuted)
        np.testing.assert_allclose(shared, self.shared)
        np.testing.assert_allclose(contrasts[0], self.contrasts[2])
        np.testing.assert_allclose(contrasts[1], self.contrasts[0])
        np.testing.assert_allclose(contrasts[2], self.contrasts[1])

    def test_rejects_incomplete_pseudo_triplet(self):
        with self.assertRaises(ValueError):
            coding.centered_terms(self.terms[:2])

    def test_type_and_common_logit_axes_are_live_and_separable(self):
        native = np.array([4.0, 1.0, 1.0])
        type0, common0 = coding.closer_type_and_common_axes(native, (0, 1, 2), 0)
        common_shift = native + 3.0
        type1, common1 = coding.closer_type_and_common_axes(common_shift, (0, 1, 2), 0)
        self.assertAlmostEqual(type1, type0)
        self.assertAlmostEqual(common1 - common0, 3.0)
        contrast_shift = native + np.array([-2.0, 1.0, 1.0])
        type2, common2 = coding.closer_type_and_common_axes(contrast_shift, (0, 1, 2), 0)
        self.assertNotEqual(type2, type0)
        self.assertAlmostEqual(common2, common0)


if __name__ == "__main__":
    unittest.main()
