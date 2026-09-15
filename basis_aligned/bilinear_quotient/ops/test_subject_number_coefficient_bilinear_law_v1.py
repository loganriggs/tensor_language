import json
import unittest

import run_subject_number_coefficient_bilinear_law_v1 as subject


class SubjectNumberCoefficientBilinearLawV1Test(unittest.TestCase):
    def test_frozen_law_reconstructs_declared_coefficients(self):
        law = json.loads(subject.LAW.read_text())
        beta = law["beta"]
        for key, value in law["predicted_coefficients"].items():
            direction, cardinality = key.split(".cardinality_")
            d = law["direction_encoding"][direction]
            c = int(cardinality)
            predicted = beta[0] + beta[1] * d + beta[2] * c + beta[3] * d * c
            self.assertAlmostEqual(predicted, value, places=10)
        self.assertLess(law["coefficient_relative_l2_error"], .023)

    def test_design_is_bound_and_model_free(self):
        plan = subject.compile_plan()
        self.assertEqual(plan["methods"], ["base", "exact", "rank1", "law"])
        self.assertEqual(plan["storage"]["coefficient_scalars_after"], 4)
        self.assertEqual(plan["price"], subject.derive_price())
        self.assertFalse(plan["model_loaded"])

    def test_patch_shape_contract(self):
        rows = [{"row_id": "x"}, {"row_id": "y"}]
        heads = {(i, subset, method): [i, len(subset), len(method)] for i in range(2)
                 for subset in subject.SUBSETS for method in subject.METHODS}
        class Torch:
            long = None
            bool = None
            @staticmethod
            def tensor(x, **_): return Fake(x)
            @staticmethod
            def full_like(x, value): return Fake([value] * len(x.value))
            @staticmethod
            def stack(x): return x
            @staticmethod
            def zeros(size, **_): return Fake([False] * size)
        class Fake:
            def __init__(self, value): self.value = value
            def __len__(self): return len(self.value)
        tokens = Fake(list(range(10)))
        tokens.device = "cpu"
        tokens.__class__.__getitem__ = lambda self, index: Fake([self.value[i] for i in index.value] if isinstance(index, Fake) else self.value[index])
        patch = subject.compile_patch(tokens, heads, rows, Torch)
        self.assertEqual(len(patch["specs"]), 2 * 16 * 4)


if __name__ == "__main__":
    unittest.main()
