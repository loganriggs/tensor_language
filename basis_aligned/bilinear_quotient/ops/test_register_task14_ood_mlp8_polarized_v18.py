#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_ood_mlp8_polarized_v18 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, fragment: str) -> dict:
    return next(x for x in plan["events"] if fragment in x["event_id"])


def metrics(item: dict) -> dict:
    return {x["name"]: x["estimate"] for x in item["metrics"]}


def test_all_registered_events_held_with_honest_data_status() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v18"
    assert len(plan["events"]) == 8
    assert all(e["verdict"] == "held" and e["evaluation_role"] == publish.EVAL_ROLE
               for e in plan["events"])
    capability = event(plan, "scoped_capability")
    assert metrics(capability)["minimum_cell_accuracy"] == 1.0
    assert "previously open" in capability["notes"]
    assert "not pristine held-out OOD" in capability["notes"]
    removal = event(plan, "selective_removal_direction_pattern")
    assert metrics(removal)["independent_replication"] is False
    assert "dependent evidence" in removal["notes"]


def test_signed_recovery_ranges_are_recomputed() -> None:
    plan = publish.build_plan()
    ps = metrics(event(plan, "plural_to_singular_signed_split"))
    assert ps["cross_recovery_range"] == pytest.approx([2.6205564693, 2.6923704406])
    assert ps["quadratic_recovery_range"] == pytest.approx([-1.3166813729, -1.2560768101])
    sp = metrics(event(plan, "singular_to_plural_signed_split"))
    assert sp["cross_recovery_range"] == pytest.approx([-0.4642656270, -0.4186317341])
    assert sp["quadratic_recovery_range"] == pytest.approx([1.4710205065, 1.4922174589])
    stable = metrics(event(plan, "background_stability"))
    assert stable["background_difference_margin_range"] == pytest.approx([0.0090602916, 0.0579230899])
    assert stable["background_difference_CE_range"] == pytest.approx([0.0082973927, 0.0681307900])


def test_scope_exclusions_and_family_are_explicit() -> None:
    plan = publish.build_plan()
    missing = plan["claim_revision"]["next_missing"]
    for phrase in ("previously open", "not pristine held-out OOD", "Independent selective removal", "necessity", "native product identities", "downstream readers"):
        assert phrase in missing
    family = plan["claim_revision"]["counterfactual_families"][-1]
    assert family["holds_fixed"][0] == "reused OOD fronted two-attractor text and subject position 8"


def test_exact_v17_prefix_and_idempotence(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    v18_present = any(c["claim_id"] == publish.NEW_CLAIM for c in before["claims"])
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
    if v18_present:
        assert after == before
    else:
        assert after["claims"][:-1] == before["claims"]
        assert after["claims"][-1]["evidence_event_ids"] == (
            before["claims"][-1]["evidence_event_ids"] + [x["event_id"] for x in plan["events"]])
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_result_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_ood_mlp8_result"]
    changed["task14_ood_mlp8_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
