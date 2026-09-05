#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_mlp8_polarized_v17 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_registered_outcomes_remain_separate_from_exploration() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v17"
    expected = {
        "instrument": "held", "cross_dominant": "null",
        "quadratic_dominant": "null", "distributed": "null",
        "downstream_interaction_needed": "null", "background_stable": "held",
        "number_specificity": "held", "lexical_collateral": "null",
        "direction_polarization_exploratory": "inconclusive",
    }
    assert len(plan["events"]) == len(expected)
    for fragment, verdict in expected.items():
        got = event(plan, fragment)
        assert got["verdict"] == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"
    exploratory = event(plan, "direction_polarization_exploratory")
    assert "not a preregistered held claim" in exploratory["notes"]


def test_direction_polarization_ranges_are_exact() -> None:
    plan = publish.build_plan()
    m = metrics(event(plan, "direction_polarization_exploratory"))
    assert m["plural_to_singular_cross_recovery_range"] == pytest.approx([2.6166451593, 2.9273193337])
    assert m["plural_to_singular_quadratic_recovery_range"] == pytest.approx([-1.5522337864, -1.4721129301])
    assert m["singular_to_plural_cross_recovery_range"] == pytest.approx([-0.6381879722, -0.3253644017])
    assert m["singular_to_plural_quadratic_recovery_range"] == pytest.approx([1.3279788942, 1.6538030538])
    stable = metrics(event(plan, "background_stable"))
    assert stable["background_recovery_difference_margin_range"] == pytest.approx([0.0004718984, 0.0524659620])
    assert stable["background_recovery_difference_CE_range"] == pytest.approx([0.0000753033, 0.0571921254])
    specificity = metrics(event(plan, "number_specificity"))
    assert specificity["lexical_margin_ratio_range"] == pytest.approx([0.0059297139, 0.0965843049])
    assert specificity["lexical_CE_ratio_range"] == pytest.approx([0.0090261280, 0.0420377521])


def test_scope_and_interpretation_limits_are_explicit() -> None:
    plan = publish.build_plan()
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert family["holds_fixed"] == [
        "licensed HOLDOUT text and subject position 8", "recipient E+A+R and MLP0--3+MR",
        "chosen recipient or opposite-number background for other MLP4--10 writes",
        "recipient L11H3 p_8, cached value, and native non-subject complement",
    ]
    missing = plan["claim_revision"]["next_missing"]
    for phrase in ("exact within-MLP8 response split", "not native product identities", "OOD syntax", "necessity", "downstream readers"):
        assert phrase in missing


def test_exact_v16_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v17_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
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
    if v17_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]])
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_mlp8_polarized_v2_result"]
    changed["task14_mlp8_polarized_v2_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
