import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
PATH = ROOT / "basis_aligned" / "polynomial_causal" / "mobius.py"
SPEC = importlib.util.spec_from_file_location("polynomial_causal_mobius", PATH)
MOBIUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOBIUS)


class PolynomialCausalTests(unittest.TestCase):
    def test_mobius_round_trip(self):
        coeff = {0: 1.5, 1: 2.0, 2: -3.0, 4: 0.25,
                 3: 4.0, 5: 0.0, 6: -1.0, 7: 2.5}
        values = {}
        for mask in range(8):
            alpha = [(mask >> i) & 1 for i in range(3)]
            values[mask] = MOBIUS.evaluate_multilinear(coeff, alpha)
        recovered = MOBIUS.mobius_coefficients(values, 3)
        for mask, expected in coeff.items():
            self.assertAlmostEqual(recovered[mask], expected)

    def test_pairwise_model_predicts_unseen_triple_without_three_way_term(self):
        def function(a):
            return 2 + 3 * a[0] - a[1] + 0.5 * a[2] + 4 * a[0] * a[2]

        train = []
        for mask in range(7):
            alpha = tuple((mask >> i) & 1 for i in range(3))
            train.append((alpha, function(alpha)))
        model = MOBIUS.fit_effect_model(train, max_degree=2)
        self.assertAlmostEqual(MOBIUS.predict_effect(model, (1, 1, 1)),
                               function((1, 1, 1)))

    def test_additive_model_misses_pair_interaction(self):
        function = lambda a: 1 + a[0] + a[1] + 5 * a[0] * a[1]
        train = [((0, 0), function((0, 0))),
                 ((1, 0), function((1, 0))),
                 ((0, 1), function((0, 1)))]
        model = MOBIUS.fit_effect_model(train, max_degree=1)
        self.assertAlmostEqual(MOBIUS.predict_effect(model, (1, 1)), 3)
        self.assertEqual(function((1, 1)), 8)


if __name__ == "__main__":
    unittest.main()
