from __future__ import annotations

import json

import freeze_mlp1_split_probe_plan as freezer


def test_plan_is_deterministic_document_disjoint_and_common_horizon() -> None:
    first = freezer.build_plan()
    second = freezer.build_plan()
    assert first == second
    selection = first["selection"]
    assert selection["contexts"] == 16
    assert len(set(selection["document_ids"])) == 16
    assert len(set(selection["row_indices"])) == 16
    assert len(selection["subset_tensor_raw_sha256"]) == 64
    assert len(selection["model_input_256_raw_sha256"]) == 64
    assert selection["subset_shape"] == [16, 513]
    assert selection["model_input_shape"] == [16, 256]
    assert selection["common_injection_position"] == 128
    assert selection["future_output_positions_per_context"] == 128
    assert selection["analysis_splits"].count("primary") == 8
    assert selection["analysis_splits"].count("replication") == 8
    assert selection["promotion_context_indices"] == list(range(12))
    assert selection["diagnostic_context_indices"] == list(range(12, 16))


def test_probe_halves_are_disjoint_and_bind_different_plans() -> None:
    plan = freezer.build_plan()
    halves = plan["probe_halves"]
    assert halves["disjoint"] is True
    assert not set(halves["first"]["probe_seeds"]) & set(
        halves["second"]["probe_seeds"]
    )
    assert halves["first"]["plan_fingerprint"] != halves["second"][
        "plan_fingerprint"
    ]
    assert "32 independent" in plan["protocol"]["fisher_rule"]
    assert plan["protocol_sha256"] == freezer.SPLIT_PROBE_PROTOCOL_SHA256
    assert plan["operator"]["backward_passes_at_batch4"] == 256


def test_plan_binds_parent_geometry_and_keeps_consequence_closed() -> None:
    plan = freezer.build_plan()
    assert plan["parent_authority"]["mlp1_directions_sha256"] == (
        freezer.EXPECTED_MLP1_DIRECTIONS_SHA256
    )
    assert plan["decision"]["consequence_stage_authorized"] is False
    assert plan["prohibitions"] == {
        "raw_logits_published": False,
        "raw_responses_published": False,
        "physical_frames_published": False,
        "projectors_published": False,
        "finite_replacement_claim": False,
        "encoder_gauge_claim": False,
    }


def test_serialized_plan_exactly_equals_builder_and_self_fingerprint() -> None:
    built = freezer.build_plan()
    serialized = json.loads(freezer.OUT.read_text())
    assert serialized == built
    assert built["plan_fingerprint"] == freezer.canonical_sha256({
        key: value for key, value in built.items() if key != "plan_fingerprint"
    })
