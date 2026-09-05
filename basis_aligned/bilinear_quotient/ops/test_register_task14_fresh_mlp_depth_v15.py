#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_mlp_depth_v15 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_separate_registered_and_exploratory_outcomes() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v15"
    assert plan["claim_revision"]["supersedes"] == "grammatical_subject_number.v14"
    expected = {
        "instrument": ("held", None),
        ".G0.": ("null", "scientific_null"),
        ".G1.": ("null", "scientific_null"),
        ".G2.": ("null", "scientific_null"),
        "distributed_groups": ("null", "scientific_null"),
        "interaction_needed": ("null", "scientific_null"),
        "number_specificity": ("held", None),
        "lexical_collateral": ("null", "scientific_null"),
        "G1_G2_exploratory": ("inconclusive", None),
    }
    assert len(plan["events"]) == len(expected)
    for fragment, verdict in expected.items():
        got = event(plan, fragment)
        assert (got["verdict"], got["failure_kind"]) == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"
    exploratory = event(plan, "G1_G2_exploratory")
    assert "not registered" in exploratory["notes"]
    assert "retroactive held claim" in exploratory["notes"]


def test_precise_ranges_are_recomputed() -> None:
    plan = publish.build_plan()
    g0 = metrics(event(plan, ".G0."))
    g1 = metrics(event(plan, ".G1."))
    g2 = metrics(event(plan, ".G2."))
    interaction = metrics(event(plan, "interaction_needed"))
    specificity = metrics(event(plan, "number_specificity"))
    exploratory = metrics(event(plan, "G1_G2_exploratory"))
    assert g0["G0_margin_recovery_range"] == pytest.approx([0.0423993862, 0.0576180152])
    assert g0["G0_CE_recovery_range"] == pytest.approx([0.0387796031, 0.0590783235])
    assert g1["G1_margin_recovery_range"] == pytest.approx([0.2174501830, 0.2555949777])
    assert g1["G1_CE_recovery_range"] == pytest.approx([0.2221435223, 0.2678423679])
    assert g2["G2_margin_recovery_range"] == pytest.approx([0.6607765856, 0.7092974517])
    assert g2["G2_CE_recovery_range"] == pytest.approx([0.6568703609, 0.7069709865])
    assert interaction["total_interaction_margin_recovery_range"] == pytest.approx([0.0209527056, 0.0547232499])
    assert interaction["total_interaction_CE_recovery_range"] == pytest.approx([0.0162089477, 0.0478027818])
    assert specificity["maximum_lexical_margin_ratio"] == pytest.approx(0.0766912252)
    assert specificity["maximum_lexical_CE_ratio"] == pytest.approx(0.0255936288)
    assert exploratory["G1_G2_margin_recovery_range"] == pytest.approx([0.9356684834, 0.9522892398])
    assert exploratory["G1_G2_CE_recovery_range"] == pytest.approx([0.9376813322, 0.9516673489])


def test_scope_is_explicitly_limited() -> None:
    plan = publish.build_plan()
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert family["holds_fixed"] == [
        "licensed HOLDOUT text",
        "recipient embedding/skip, accumulated attention, and state remainder",
        "recipient subject score p_8",
        "recipient cached value and native non-subject complement",
    ]
    missing = plan["claim_revision"]["next_missing"]
    for phrase in ("mostly additive MLP4--10", "individual-MLP", "Necessity", "OOD syntax", "downstream readers"):
        assert phrase in missing


def test_exact_v14_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v15_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
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
    if v15_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]])
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp_depth_result"]
    changed["task14_mlp_depth_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
