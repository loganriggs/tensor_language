#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_matched_qk_v11 as publish


RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def by_suffix(plan: dict, suffix: str) -> dict:
    return next(event for event in plan["events"] if event["event_id"].endswith(suffix))


def test_plan_keeps_the_independent_outcomes_separate() -> None:
    plan = publish.build_plan()
    assert len(plan["artifacts"]) == 11
    assert len(plan["events"]) == 8
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v11"
    assert plan["claim_revision"]["supersedes"] == "grammatical_subject_number.v10"
    assert plan["claim_revision"]["status"] == "site_live"
    verdicts = {event["event_id"]: (event["verdict"], event["failure_kind"])
                for event in plan["events"]}
    assert verdicts["task14_head11_3.fresh_matched_qk.v3_instrument.complete.v1"] == ("held", None)
    assert verdicts["task14_head11_3.fresh_matched_qk.number_discrimination.complete.v1"] == ("null", "scientific_null")
    assert verdicts["task14_head11_3.fresh_matched_qk.lexical_selectivity.complete.v1"] == ("null", "scientific_null")
    assert verdicts["task14_head11_3.fresh_matched_qk.bidirectional_task_use.complete.v1"] == ("null", "scientific_null")
    assert verdicts["task14_head11_3.fresh_matched_qk.directional_asymmetry.complete.v1"] == ("held", None)
    assert "must not be repeated" in plan["claim_revision"]["next_missing"]


def test_decisive_metrics_are_recomputed_from_v3_cells() -> None:
    plan = publish.build_plan()
    number = {m["name"]: m["estimate"] for m in
              by_suffix(plan, "number_discrimination.complete.v1")["metrics"]}
    lexical = {m["name"]: m["estimate"] for m in
               by_suffix(plan, "lexical_selectivity.complete.v1")["metrics"]}
    use = {m["name"]: m["estimate"] for m in
           by_suffix(plan, "bidirectional_task_use.complete.v1")["metrics"]}
    asym = {m["name"]: m["estimate"] for m in
            by_suffix(plan, "directional_asymmetry.complete.v1")["metrics"]}
    assert number["minimum_number_score_absolute_mean_margin_effect"] == pytest.approx(0.09524405002593994)
    assert number["minimum_expected_row_sign_fraction"] == 0.0
    assert number["cells_passing_both_value_state_signs"] == 2
    assert lexical["maximum_same_number_lexical_over_number_margin_ratio"] == pytest.approx(0.37623193672617516)
    assert lexical["cells_passing_both_value_state_lexical_ratios"] == 1
    assert use["direction_template_cells_passing_task_use"] == 2
    assert asym["singular_to_plural_mean_donor_margin_improvement_range"] == pytest.approx([0.30822837352752686, 0.40423595905303955])
    assert asym["plural_to_singular_mean_donor_margin_improvement_range"] == pytest.approx([-0.13620710372924805, -0.09524405002593994])


def test_invalid_lineage_is_provenance_not_scientific_evidence() -> None:
    plan = publish.build_plan()
    v1 = by_suffix(plan, "v1_engineering_failure.invalid.v1")
    v2 = by_suffix(plan, "v2_instrument.invalid.v1")
    v3 = by_suffix(plan, "v3_instrument.complete.v1")
    assert v1["failure_kind"] == "implementation_failure"
    assert "no scientific outcome" in v1["notes"]
    assert v2["failure_kind"] == "invalid_instrument"
    assert "descriptive only" in v2["notes"]
    assert v2["supersedes_event_id"] == v1["event_id"]
    assert v3["supersedes_event_id"] == v2["event_id"]


def test_literal_hash_or_outcome_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_fresh_matched_qk_v3_result"]
    changed["task14_fresh_matched_qk_v3_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()


def test_apply_is_idempotent_and_event_ids_are_global_unique(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
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
    record = json.loads(first)
    assert record["claims"][-1]["claim_id"] == "grammatical_subject_number.v11"
    ids = [event["event_id"] for event in record["evidence_events"]]
    assert len(ids) == len(set(ids))
    assert record["claims"][-1]["evidence_event_ids"] == (
        record["claims"][-2]["evidence_event_ids"] +
        [event["event_id"] for event in plan["events"]]
    )
