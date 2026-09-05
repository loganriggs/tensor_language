#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_mlp4_10_conditional_v16 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_global_and_layer_outcomes_are_separate() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v16"
    expected = {
        "instrument": "held", "at_least_one_standalone": "held",
        "at_least_one_conditional": "held", "same_layer_stable": "held",
        "context_dependence": "null", "number_specificity": "held",
        "lexical_collateral": "null",
    }
    for fragment, verdict in expected.items():
        assert event(plan, fragment)["verdict"] == verdict
    for layer in range(4, 11):
        got = event(plan, f".MLP{layer}.")
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"
        assert got["verdict"] == ("held" if layer == 8 else "inconclusive" if layer == 10 else "null")


def test_exact_layer_ranges_are_recomputed() -> None:
    plan = publish.build_plan()
    m8 = metrics(event(plan, ".MLP8."))
    assert m8["MLP8_standalone_margin_recovery_range"] == pytest.approx([0.2711236020, 0.2768010769])
    assert m8["MLP8_standalone_CE_recovery_range"] == pytest.approx([0.2655728276, 0.2826940510])
    assert m8["MLP8_conditional_margin_recovery_range"] == pytest.approx([0.2864888089, 0.3259935226])
    assert m8["MLP8_conditional_CE_recovery_range"] == pytest.approx([0.2802380304, 0.3120196497])
    m10 = metrics(event(plan, ".MLP10."))
    assert m10["MLP10_standalone_margin_recovery_range"] == pytest.approx([0.2187541561, 0.2654210120])
    assert m10["MLP10_conditional_CE_recovery_range"] == pytest.approx([0.2223353233, 0.3044318310])
    specificity = metrics(event(plan, "number_specificity"))
    assert specificity["maximum_lexical_margin_ratio"] == pytest.approx(0.0792046478)
    assert specificity["maximum_lexical_CE_ratio"] == pytest.approx(0.0240415517)


def test_scope_and_boundary_limit_are_explicit() -> None:
    plan = publish.build_plan()
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert family["holds_fixed"] == [
        "licensed HOLDOUT text", "recipient embedding/skip, attention, and state remainder",
        "recipient MLP0--3 plus grouping remainder",
        "recipient p_8, cached value, and native non-subject complement",
    ]
    missing = plan["claim_revision"]["next_missing"]
    for phrase in ("only native-layer handle", "not a claim that all of MLP8", "Within-MLP", "necessity", "OOD syntax", "downstream readers"):
        assert phrase in missing


def test_exact_v15_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v16_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
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
    if v16_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]])
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp4_10_conditional_result"]
    changed["task14_mlp4_10_conditional_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
