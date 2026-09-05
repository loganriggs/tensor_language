#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_mlp8_late_writer_halves_v21 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_all_seven_registered_outcomes_are_separate() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v21"
    expected = {
        "instrument_parent_closure": "held",
        ".X_mlp6_7_dominant": "null",
        ".W_mlp4_5_dominant": "null",
        ".distributed_within_V": "held",
        ".WX_composition": "null",
        ".direction_switch": "null",
        ".number_specificity": "held",
    }
    assert len(plan["events"]) == len(expected)
    for fragment, verdict in expected.items():
        got = event(plan, fragment)
        assert got["verdict"] == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"


def test_quantitative_ranges_are_recomputed() -> None:
    plan = publish.build_plan()
    x = metrics(event(plan, ".X_mlp6_7_dominant"))
    assert x["X_aggregate_recovery_range"] == pytest.approx([0.6262675126569943, 0.9681843016933782])
    assert x["W_only_aggregate_recovery_range"] == pytest.approx([0.031815698306621835, 0.3737324873430057])
    w = metrics(event(plan, ".W_mlp4_5_dominant"))
    assert w["W_aggregate_recovery_range"] == pytest.approx([0.027235651844973844, 0.6675329390460267])
    assert w["X_only_aggregate_recovery_range"] == pytest.approx([0.33246706095397327, 0.9727643481550261])
    assert metrics(event(plan, ".WX_composition"))["WX_interaction_recovery_range"] == pytest.approx(
        [-0.22904543338104033, 0.49614802148853876]
    )
    assert metrics(event(plan, ".number_specificity"))["maximum_lexical_ratio"] == pytest.approx(
        0.233795410169825
    )


def test_no_dominant_half_and_operational_scope_are_explicit() -> None:
    plan = publish.build_plan()
    assert metrics(event(plan, ".direction_switch"))["direction_winners"] == {
        "plural_to_singular": None,
        "singular_to_plural": None,
    }
    missing = plan["claim_revision"]["next_missing"]
    for phrase in (
        "neither half dominant", "distributed contribution", "operational native writer groups",
        "not unique semantic units", "Individual MLP4--7 identities", "OOD replication",
        "independent data", "downstream readers", "necessity",
    ):
        assert phrase in missing


def test_exact_v20_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v21_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
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
    if v21_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]]
        )
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp8_late_writer_halves_result"]
    changed["task14_mlp8_late_writer_halves_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
