#!/usr/bin/env python3
# BQLANE: cpu
"""Tests for machine-checked circuit prior-art receipts."""

from __future__ import annotations

import copy
import hashlib
import unittest

import circuit_prior_art as gate


def valid_receipt() -> dict:
    return {
        "schema": gate.SCHEMA,
        "candidate_id": "task14.subject_number_state",
        "canonical_objects": ["behavior.subject_verb.number_agreement"],
        "aliases_searched": ["subject verb agreement", "SVA", "agreement attractor"],
        "method_families": ["task_conditioned_causal_direction", "DAS"],
        "matched_prior_claims": ["circuits.module_dossiers.subject_verb_agreement"],
        "relation": "extension",
        "novelty_delta": "Test complete-subject state transfer across syntax and noun identity.",
        "decision_changed": "Determines whether one causal state can replace the agreement computation.",
        "reviewer": "independent-reviewer",
        "reviewed_sources": [
            {
                "source": "circuits/MODULE_DOSSIERS.md",
                "sha256": "12" * 32,
                "searched_terms": ["subject verb", "agreement", "attractor"],
            },
            {
                "source": "circuits/METHOD_FAILURES.md",
                "sha256": "34" * 32,
                "searched_terms": ["DAS", "causal direction"],
            },
        ],
    }


class PriorArtReceiptTests(unittest.TestCase):
    def test_valid_receipt_has_stable_canonical_hash(self) -> None:
        receipt = valid_receipt()
        digest = gate.validate_receipt(receipt)
        self.assertEqual(digest, gate.canonical_hash(receipt))
        reordered = {key: receipt[key] for key in reversed(receipt)}
        self.assertEqual(gate.validate_receipt(reordered), digest)

    def test_new_question_requires_no_matches_and_other_relations_require_matches(self) -> None:
        receipt = valid_receipt()
        receipt["relation"] = "new_question"
        with self.assertRaisesRegex(gate.PriorArtError, "cannot have"):
            gate.validate_receipt(receipt)
        receipt["matched_prior_claims"] = []
        gate.validate_receipt(receipt)
        for relation in ("replication", "extension", "contradiction_test"):
            receipt["relation"] = relation
            with self.assertRaisesRegex(gate.PriorArtError, "requires a prior match"):
                gate.validate_receipt(receipt)

    def test_required_search_and_evidence_fields_cannot_be_empty(self) -> None:
        for field in (
            "canonical_objects", "aliases_searched", "method_families", "novelty_delta",
            "decision_changed", "reviewer", "reviewed_sources",
        ):
            receipt = valid_receipt()
            receipt[field] = [] if isinstance(receipt[field], list) else ""
            with self.subTest(field=field), self.assertRaises(gate.PriorArtError):
                gate.validate_receipt(receipt)
        receipt = valid_receipt()
        receipt["reviewed_sources"][0]["searched_terms"] = []
        with self.assertRaisesRegex(gate.PriorArtError, "searched_terms"):
            gate.validate_receipt(receipt)

    def test_sources_require_exact_sha_and_searched_terms(self) -> None:
        for mutation in ("ABC", "gg" * 32, "AB" * 32):
            receipt = valid_receipt()
            receipt["reviewed_sources"][0]["sha256"] = mutation
            with self.subTest(mutation=mutation), self.assertRaises(gate.PriorArtError):
                gate.validate_receipt(receipt)
        receipt = valid_receipt()
        del receipt["reviewed_sources"][0]["searched_terms"]
        with self.assertRaisesRegex(gate.PriorArtError, "fields"):
            gate.validate_receipt(receipt)

    def test_duplicate_evidence_and_text_values_reject(self) -> None:
        receipt = valid_receipt()
        receipt["aliases_searched"].append(receipt["aliases_searched"][0])
        with self.assertRaisesRegex(gate.PriorArtError, "duplicates"):
            gate.validate_receipt(receipt)
        receipt = valid_receipt()
        receipt["reviewed_sources"].append(copy.deepcopy(receipt["reviewed_sources"][0]))
        with self.assertRaisesRegex(gate.PriorArtError, "duplicate source"):
            gate.validate_receipt(receipt)

    def test_list_rejects_duplicate_candidate_ids(self) -> None:
        first = valid_receipt()
        second = copy.deepcopy(first)
        second["novelty_delta"] = "A different proposed delta cannot reuse the candidate identity."
        with self.assertRaisesRegex(gate.PriorArtError, "duplicate candidate_id"):
            gate.validate_receipts([first, second])
        second["candidate_id"] = "task14.subject_number_reader"
        self.assertEqual(len(gate.validate_receipts([first, second])), 2)

    def test_unknown_or_missing_fields_and_nonfinite_json_reject(self) -> None:
        receipt = valid_receipt()
        receipt["extra"] = "not allowed"
        with self.assertRaisesRegex(gate.PriorArtError, "fields"):
            gate.validate_receipt(receipt)
        receipt = valid_receipt()
        with self.assertRaisesRegex(gate.PriorArtError, "finite JSON"):
            gate.canonical_hash({"unexpected_number": float("nan")})

    def test_reviewed_sources_must_match_current_contained_files(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "MODULE_DOSSIERS.md"
            source.write_text("known circuit evidence\n")
            receipt = valid_receipt()
            receipt["reviewed_sources"] = [{
                "source": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "searched_terms": ["known circuit"],
            }]
            self.assertEqual(
                gate.validate_source_files(receipt, root), gate.canonical_hash(receipt),
            )
            source.write_text("changed evidence\n")
            with self.assertRaisesRegex(gate.PriorArtError, "source changed"):
                gate.validate_source_files(receipt, root)
            receipt["reviewed_sources"][0]["source"] = "../outside.md"
            with self.assertRaisesRegex(gate.PriorArtError, "not contained"):
                gate.validate_source_files(receipt, root)


if __name__ == "__main__":
    unittest.main()
