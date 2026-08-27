import json
import unittest
from pathlib import Path

import torch

from .shared_value_gauge import apply_shared_gauge, value_output_action
from .shared_value_output_codec import (decode_shared_head, descriptive_bits,
                                        encode_shared_head)


HERE = Path(__file__).resolve().parent


class SharedValueOutputCodecTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(451)
        self.values = [torch.randn(3, 7, generator=generator, dtype=torch.float64)
                       for _ in range(4)]
        self.outputs = [torch.randn(7, 3, generator=generator, dtype=torch.float64)
                        for _ in range(4)]
        self.locals = [torch.randn(7, 5, generator=generator, dtype=torch.float64)
                       for _ in range(4)]
        self.shared = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        self.mixing = [0.0, 0.2, 0.6, 0.9]
        self.gauge = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        self.gauge += 4 * torch.eye(3, dtype=torch.float64)

    def test_shared_gauge_has_identical_canonical_bytes(self):
        first = encode_shared_head(self.values, self.outputs, 14)
        values, outputs = apply_shared_gauge(self.values, self.outputs, self.gauge)
        second = encode_shared_head(values, outputs, 14)
        self.assertEqual(first, second)

    def test_decode_round_trip_has_small_action_error(self):
        encoded = encode_shared_head(self.values, self.outputs, 18)
        header, values, outputs = decode_shared_head(encoded)
        expected = value_output_action(self.values, self.outputs, self.locals,
                                       self.shared, self.mixing)
        actual = value_output_action(values, outputs, self.locals,
                                     self.shared, self.mixing)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=2e-4, rtol=2e-4)
        self.assertEqual(header["tensor_order"],
                         ["V0", "V1", "V2", "V3", "O0", "O1", "O2", "O3"])
        self.assertEqual(descriptive_bits(encoded), 8 * len(encoded))

    def test_independent_layer_gauge_does_not_collapse_to_same_bytes(self):
        values = list(self.values); outputs = list(self.outputs)
        values[2] = self.gauge @ values[2]
        outputs[2] = outputs[2] @ torch.linalg.inv(self.gauge)
        self.assertNotEqual(encode_shared_head(self.values, self.outputs, 14),
                            encode_shared_head(values, outputs, 14))

    def test_rejects_singular_parent_bad_precision_and_malformed_payload(self):
        singular = list(self.values); singular[0] = torch.ones_like(singular[0])
        with self.assertRaises(ValueError):
            encode_shared_head(singular, self.outputs)
        with self.assertRaises(ValueError):
            encode_shared_head(self.values, self.outputs, 25)
        encoded = encode_shared_head(self.values, self.outputs)
        with self.assertRaises(ValueError):
            decode_shared_head(encoded[:-1])

    def test_contract_refuses_promotion_and_names_missing_head_gauge(self):
        contract = json.loads((HERE / "shared_value_output_codec_contract.json").read_text())
        self.assertIn("not_checkpoint_priced", contract["status"])
        self.assertIn("S9", contract["scope"])
        self.assertTrue(contract["encoding"]["literal_bits_are_descriptive_only"])
        self.assertIn("held-out, composite, extraction, removal, and OOD",
                      contract["required_tests"][-1])


if __name__ == "__main__":
    unittest.main()
