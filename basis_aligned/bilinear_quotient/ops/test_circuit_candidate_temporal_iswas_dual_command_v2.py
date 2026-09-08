#!/usr/bin/env python3
import unittest

import circuit_candidate_temporal_iswas_dual_command_v2 as candidate


class PrefixPreservedDualCommandTests(unittest.TestCase):
    def test_authority(self):
        rows = candidate.build_rows()
        self.assertEqual(len(rows), 32)
        self.assertEqual(candidate.validate_rows(rows), candidate.EXPECTED_AUTHORITY_SHA256)

    def test_complete_factorial_and_positions(self):
        for row in candidate.build_rows():
            self.assertEqual(set(row["endpoints"]), set(candidate.CELLS))
            for endpoint in row["endpoints"].values():
                self.assertEqual(endpoint["ids"][endpoint["temporal_position"]],
                                 endpoint["temporal_answer_id"])
                self.assertEqual(endpoint["ids"][endpoint["iswas_position"]],
                                 endpoint["iswas_answer_id"])

    def test_temporal_logits_have_identical_qualified_prefixes(self):
        temporal_rows = candidate.temporal.build_rows()
        source = {row["row_id"]: row for row in temporal_rows}
        for row in candidate.build_rows():
            original = source[row["temporal_source_row_id"]]
            for endpoint in row["endpoints"].values():
                answer = " will" if endpoint["temporal_answer_id"] == candidate._single(" will") else " had"
                self.assertEqual(endpoint["temporal_prefix"], candidate._side(original, answer))
                self.assertEqual(endpoint["temporal_prefix_ids"],
                                 candidate.ENCODING.encode(candidate._side(original, answer)))


if __name__ == "__main__": unittest.main()
