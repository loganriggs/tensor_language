import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = json.loads((HERE / "shared_value_gauge_audit_results.json").read_text())
SOURCE = (HERE / "shared_value_gauge_audit.py").read_text()


class SharedValueGaugeAuditResultTest(unittest.TestCase):
    def test_checkpoint_and_revision_are_frozen(self):
        self.assertEqual(RESULT["revision"],
                         "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240")
        self.assertEqual(len(RESULT["checkpoint_sha256"]), 64)
        self.assertIn("local_files_only=True", SOURCE)

    def test_every_shared_edge_is_nonzero(self):
        coefficients = RESULT["attention_value_mixing_coefficients"]
        self.assertEqual(len(coefficients), 18)
        self.assertTrue(all(value != 0 for value in coefficients))
        self.assertTrue(RESULT["all_layers_have_nonzero_shared_value_edge"])

    def test_every_shared_head_is_well_inside_full_rank_stratum(self):
        heads = RESULT["layer0_shared_value_heads"]
        self.assertEqual(len(heads), 9)
        self.assertTrue(all(row["rank_at_relative_tolerance_1e_7"] == 128
                            for row in heads))
        self.assertGreater(RESULT["minimum_relative_singular_value"], .39)
        self.assertTrue(RESULT["all_shared_value_heads_full_row_rank"])


if __name__ == "__main__":
    unittest.main()
