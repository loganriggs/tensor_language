#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_current_cached_value_v13 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, suffix: str) -> dict:
    return next(x for x in plan["events"] if x["event_id"].endswith(suffix))


def test_outcomes_are_separate_and_scope_is_exact() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v13"
    assert plan["claim_revision"]["supersedes"] == "grammatical_subject_number.v12"
    assert plan["claim_revision"]["status"] == "site_live"
    assert len(plan["events"]) == 7
    expected = {
        "v1_instrument.invalid.v1": ("invalid", "implementation_failure"),
        "v2_instrument.complete.v1": ("held", None),
        "current_branch_use.complete.v1": ("held", None),
        "cached_branch_use.complete.v1": ("null", "scientific_null"),
        "interaction_needed.complete.v1": ("null", "scientific_null"),
        "lexical_leakage.complete.v1": ("null", "scientific_null"),
        "number_specificity.complete.v1": ("held", None),
    }
    for suffix, verdict in expected.items():
        got = event(plan, suffix)
        assert (got["verdict"], got["failure_kind"]) == verdict
        assert got["evaluation_role"] == "FRESH_LICENSED_HOLDOUT"
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert "recipient subject score p_8" in family["holds_fixed"]
    assert "native non-subject source-term complement" in family["holds_fixed"]
    assert "Necessity/selective removal" in plan["claim_revision"]["next_missing"]


def test_decisive_metrics_are_recomputed() -> None:
    plan = publish.build_plan()
    current = {m["name"]: m["estimate"] for m in event(plan, "current_branch_use.complete.v1")["metrics"]}
    cached = {m["name"]: m["estimate"] for m in event(plan, "cached_branch_use.complete.v1")["metrics"]}
    interaction = {m["name"]: m["estimate"] for m in event(plan, "interaction_needed.complete.v1")["metrics"]}
    specificity = {m["name"]: m["estimate"] for m in event(plan, "number_specificity.complete.v1")["metrics"]}
    assert current["current_mean_margin_range"] == pytest.approx([0.2865447998, 0.5899844170])
    assert current["current_over_joint_margin_range"] == pytest.approx([1.1203790838, 1.2155608407])
    assert current["minimum_current_positive_row_fraction"] == .75
    assert cached["cached_mean_margin_range"] == pytest.approx([-0.0654420853, -0.0376197100])
    assert cached["maximum_cached_positive_row_fraction"] == .25
    assert interaction["interaction_mean_margin_range"] == pytest.approx([-0.0257711411, -0.0042405128])
    assert specificity["maximum_lexical_over_number_joint_margin_ratio"] == pytest.approx(0.1399281196)


def test_v1_is_engineering_only_and_v2_supersedes_it() -> None:
    plan = publish.build_plan()
    invalid = event(plan, "v1_instrument.invalid.v1")
    valid = event(plan, "v2_instrument.complete.v1")
    assert "not evidence" in invalid["notes"]
    assert valid["supersedes_event_id"] == invalid["event_id"]


def test_exact_v12_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    # The live record may now contain revisions after v13. An old publisher
    # must remain an exact no-op there rather than assuming its revision is
    # permanently the tail of the claim list.
    v13_already_present = any(
        claim["claim_id"] == publish.NEW_CLAIM for claim in before["claims"])
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
    if v13_already_present:
        assert after == before
        v13 = next(c for c in after["claims"] if c["claim_id"] == publish.NEW_CLAIM)
        assert v13["evidence_event_ids"][-len(plan["events"]):] == [
            x["event_id"] for x in plan["events"]]
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"]
            + [x["event_id"] for x in plan["events"]]
        )
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_current_cached_v2_result"]
    changed["task14_current_cached_v2_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
