#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_mlp8_depth_sources_v20 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_invalid_attempt_and_all_seven_predictions_are_separate() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v20"
    expected = {
        "numeric_grouping_attempt": "invalid",
        "instrument_parent_closure": "held",
        ".V_late_dominant": "held",
        ".U_early_dominant": "null",
        ".distributed_depth": "null",
        ".cross_depth_composition": "null",
        ".direction_switch": "null",
        ".number_specificity": "held",
    }
    assert len(plan["events"]) == len(expected)
    for fragment, verdict in expected.items():
        got = event(plan, fragment)
        assert got["verdict"] == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"

    invalid = event(plan, "numeric_grouping_attempt")
    assert invalid["stage"] == "invalid"
    assert invalid["failure_kind"] == "implementation_failure"
    assert invalid["result_artifact_id"] == "task14_mlp8_depth_sources_invalid_attempt"
    assert "scientific arms" in invalid["notes"]
    assert invalid["supersedes_event_id"] is None
    assert event(plan, "instrument_parent_closure")["supersedes_event_id"] == invalid["event_id"]


def test_quantitative_ranges_are_recomputed_from_valid_result() -> None:
    plan = publish.build_plan()
    late = metrics(event(plan, ".V_late_dominant"))
    assert late["V_aggregate_recovery_range"] == pytest.approx([0.75444302446817, 1.0954158055033747])
    assert late["U_only_aggregate_recovery_range"] == pytest.approx([-0.09541580550337483, 0.24555697553182998])

    early = metrics(event(plan, ".U_early_dominant"))
    assert early["U_aggregate_recovery_range"] == pytest.approx([0.07210828830081623, 0.48400315048658205])
    assert early["V_only_aggregate_recovery_range"] == pytest.approx([0.5159968495134181, 0.9278917116991838])

    interaction = metrics(event(plan, ".cross_depth_composition"))
    assert interaction["absolute_UV_interaction_recovery_range"] == pytest.approx(
        [1.8303125746624272e-05, 0.5793894392370365]
    )
    assert metrics(event(plan, ".number_specificity"))["maximum_lexical_ratio"] == pytest.approx(
        0.18773508344800818
    )
    assert metrics(event(plan, ".direction_switch"))["direction_winners"] == {
        "plural_to_singular": "V",
        "singular_to_plural": "V",
    }


def test_invalid_attempt_preserves_exact_failed_closure_values() -> None:
    invalid = metrics(event(publish.build_plan(), "numeric_grouping_attempt"))
    assert invalid == pytest.approx({
        "input_state_closure_max_absolute_error": 0.000244140625,
        "parent_raw_state_max_absolute_error": 0.00048828125,
        "parent_MLP8_output_max_absolute_error": 9.104936071935299e-05,
        "parent_propagated_slot_max_absolute_error": 7.62939453125e-05,
    })


def test_scope_and_operational_group_limit_are_explicit() -> None:
    plan = publish.build_plan()
    missing = plan["claim_revision"]["next_missing"]
    for phrase in (
        "operational native writer groups", "not a unique semantic basis",
        "Individual MLP4--7 identities", "within-group semantic units",
        "OOD replication", "independent data", "downstream readers", "necessity",
    ):
        assert phrase in missing
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert family["holds_fixed"] == [
        "licensed HOLDOUT text and subject position 8",
        "recipient other MLP4--10 downstream background",
        "recipient L11H3 p_8 and cached value",
        "native non-subject L11H3 source complement",
    ]


def test_exact_v19_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v20_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    copied = circuits / RECORD.name
    shutil.copy2(RECORD, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    publish.apply_plan(plan, regenerate=False)
    first = copied.read_bytes()
    publish.apply_plan(plan, regenerate=False)
    assert copied.read_bytes() == first
    after = json.loads(first)
    if v20_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]]
        )
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_valid_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp8_depth_sources_result"]
    changed["task14_mlp8_depth_sources_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
