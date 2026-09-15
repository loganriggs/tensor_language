import json
import math
import unittest

import run_subject_number_native_weight_axis_v1 as subject


class SubjectNumberNativeWeightAxisV1Test(unittest.TestCase):
    def test_weight_axis_is_unit_and_outcome_blind(self):
        axis = json.loads(subject.NATIVE_AXIS.read_text())
        self.assertEqual(axis["construction"], "top_left_singular_vector_of_native_head_output_projection")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in axis["axis"])), 1.0, places=6)
        self.assertGreater(axis["prior_axis_cosine"], .94)
        self.assertFalse(axis["causal_outcomes_read"])
        self.assertFalse(axis["behavior_fit"])
        self.assertFalse(axis["axis_combination_fit"])

    def test_bound_model_free_plan(self):
        plan = subject.compile_plan()
        self.assertEqual(plan["methods"], ["base", "exact", "law", "native_axis"])
        self.assertEqual(plan["price"], subject.derive_price())
        self.assertFalse(plan["model_loaded"])
        self.assertGreater(plan["native_axis_prior_cosine"], .94)

    def test_prediction_registry_is_closed(self):
        self.assertEqual(len(subject.PREDICTION_REGISTRY), 5)
        self.assertTrue(all(value is None for value in subject.PREDICTION_REGISTRY.values()))


if __name__ == "__main__": unittest.main()
