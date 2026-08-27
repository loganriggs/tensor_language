import json
import unittest
from pathlib import Path

from .shared_value_gauge import generic_parameter_dimension

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE / "shared_value_output_quotient_contract.json").read_text())
LEDGER = json.loads((HERE / "shared_bus_ledger.json").read_text())


class SharedValueOutputQuotientContractTest(unittest.TestCase):
    def test_dimensions_are_computed_not_freehand(self):
        architecture = CONTRACT["architecture"]
        measured = generic_parameter_dimension(
            architecture["layers"], architecture["model_dimension"],
            architecture["head_dimension"], architecture["heads"])
        for key, value in measured.items():
            self.assertEqual(CONTRACT["generic_dimension"][key], value)
        self.assertEqual(
            CONTRACT["generic_dimension"]["quotient_dimension"]-
            CONTRACT["generic_dimension"]["incorrect_independent_layer_quotient_dimension"],
            CONTRACT["generic_dimension"]["independent_layer_undercount"])

    def test_contract_refuses_to_turn_dimension_into_bits(self):
        self.assertIn("not_yet_priced", CONTRACT["status"])
        self.assertIn("fixture implements", CONTRACT["canonical_section"]["codec_status"])
        self.assertIn("QK-keyed common S9", CONTRACT["canonical_section"]["codec_status"])
        self.assertIn("quotient_bits remain absent",
                      CONTRACT["canonical_section"]["codec_status"])
        self.assertTrue(any("not a canonical bit price" in item
                            for item in CONTRACT["interpretation_limits"]))

    def test_ledger_links_exact_contract_but_keeps_price_null(self):
        value_bus = LEDGER["objects"]["shared_value_v0"]
        self.assertEqual(value_bus["value_output_quotient_contract"],
                         "shared_value_output_quotient_contract.json")
        self.assertIsNone(value_bus["quotient_bits"])
        self.assertIn("shared across all 18 layers", value_bus["exact_value_output_gauge"])

    def test_common_head_permutation_scope_is_explicit(self):
        permutation = CONTRACT["discrete_symmetry"]
        self.assertEqual(permutation["group"], "common S9 head permutation")
        self.assertTrue(permutation["not_independent_per_layer"])


if __name__ == "__main__":
    unittest.main()
