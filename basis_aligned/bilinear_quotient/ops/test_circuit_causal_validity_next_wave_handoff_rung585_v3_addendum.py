"""CPU-only schema checks for the prospective R585 next-wave v3 addendum."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


OPS = Path(__file__).resolve().parent
ROOT = OPS.parents[2]
V1 = OPS / "circuit_causal_validity_next_wave_handoff_rung585.json"
V2 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v2_addendum.json"
V3 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v3_addendum.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict[str, object]:
    return json.loads(
        V3.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ),
    )


def test_v3_is_strict_json_and_binds_immutable_v1_v2():
    packet = load()
    assert packet["schema"] == "circuit_causal_validity_next_wave_handoff_v3_addendum"
    assert packet["base_contract_sha256"] == sha256(V1) == (
        "e8970f9ef2d7eb7b291a5fb288833bc252e62fabf1016a699e981c19a6be560a"
    )
    assert packet["v2_contract_sha256"] == sha256(V2) == (
        "eb8ef7d00324c7f38210f0e8303951d97282fc8dbede9ee10ef8409db414709b"
    )
    assert packet["outcome_boundary"]["applies_prospectively"] is True
    assert packet["outcome_boundary"]["changes_frozen_scientific_thresholds"] is False


def test_exact_lessons_and_invalid_failure_contract_are_machine_readable():
    packet = load()
    assert [row["lesson"] for row in packet["accepted_lessons"]] == [19, 20, 21]
    invalid = packet["invalid_failure_contract"]
    assert invalid["canonical_order"] == (
        "ascending_lexicographic_order_of_complete_clause_strings"
    )
    assert invalid["duplicate_policy"] == "forbidden"
    assert invalid["fit"]["failure_class"] == "invalid_instrument"
    assert invalid["select"]["failure_class"] == "select_invalid_instrument"
    assert invalid["select"]["prefix_policy"].endswith("add_select_prefix_once")
    assert len(invalid["retained_recomputable_sources"]) == len(set(
        invalid["retained_recomputable_sources"]
    ))


def test_hard_abort_list_is_explicit_and_cannot_be_serialized_as_a_null():
    packet = load()
    contract = packet["hard_abort_contract"]
    expected = {
        "incomplete_factor_capture",
        "live_factorization_at_intervened_state",
        "native_full_attention_reconstruction",
        "nonfinite_model_or_evidence_tensor",
        "observed_hook_write_matches_requested_delta",
        "preregistered_structural_full_vocabulary_logit_identity",
        "replay_native_full_vocabulary_logit_identity",
    }
    assert set(contract["checks"]) == expected
    assert contract["checks"] == sorted(contract["checks"])
    assert contract["failure_behavior"] == "abort_before_any_final_namespace_publication"
    assert contract["published_clause_policy"].startswith("forbidden")


def test_live_and_saved_structural_checks_are_distinct():
    structural = load()["structural_identity_contract"]
    live = structural["live_preregistered_check"]
    saved = structural["saved_auditable_check"]
    assert live["comparison"] != saved["comparison"]
    assert live["evidence_retained"] is False
    assert saved["evidence_retained"] is True
    assert live["failure_behavior"] == "hard_abort_before_publication"
    assert saved["failure_behavior"] == "derive_canonically_ordered_phase_invalid_clause"
    assert saved["evidence_field"] == "structural_inserted_term_identity_checks"
    assert live["threshold"] == saved["threshold"] == 1e-5
    assert "does_not_discharge" in structural["no_substitution_rule"]


def test_planted_negative_and_required_test_censuses_are_complete_and_unique():
    packet = load()
    fixtures = packet["planted_negative_fixture_ids"]
    tests = packet["required_test_ids"]
    assert len(fixtures) == len(set(fixtures)) == 8
    assert len(tests) == len(set(tests)) == 9
    assert any("fit_invalid" in value for value in fixtures)
    assert any("select_invalid" in value for value in fixtures)
    assert any("equal_inserted_tensors_unequal_full_logits" in value for value in fixtures)
    assert any("saved_inserted_tensor_identity" in value for value in tests)


def test_v3_does_not_claim_model_or_outcome_work():
    packet = load()
    forbidden = set(packet["outcome_boundary"]["forbidden_actions"])
    assert "load_model_or_cuda" in forbidden
    assert "open_scientific_outcomes" in forbidden
    assert "enqueue_gpu_work" in forbidden
    encoded = V3.read_text().lower()
    assert "model_forwards" not in encoded
    assert "result_observed" not in encoded
