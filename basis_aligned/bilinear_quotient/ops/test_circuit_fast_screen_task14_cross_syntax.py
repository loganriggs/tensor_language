#!/usr/bin/env python3
"""CPU-only tests for the targeted Task 14 cross-syntax interchange."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import json
import sys

import circuit_fast_screen_candidate_task14_cross_syntax as candidate
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_task14_cross_syntax as run


def test_exact_frozen_relation_census_and_direction_balance() -> None:
    rows = candidate.build_rows()
    assert len(rows) == 64
    assert [row["donor_ordinal"] for row in rows] == list(range(832, 896))
    assert candidate.validate_rows(rows) == \
        "9ec9730ffdaa00e3ed43909cf09d355e6e42e693f7d63a0176988da78ed16b95"
    assert candidate.validate_rows(rows) == run.EXPECTED_AUTHORITY_SHA256
    cells = {cell: sum(row["cell_id"] == cell for row in rows) for cell in {
        str(row["cell_id"]) for row in rows
    }}
    assert cells == {
        "pp_plural_to_relative_singular": 16,
        "pp_singular_to_relative_plural": 16,
        "relative_plural_to_pp_singular": 16,
        "relative_singular_to_pp_plural": 16,
    }
    assert all(
        {row["target_family"], row["donor_family"]} == {"A1", "A2"}
        and row["base_subject_number"] != row["donor_subject_number"]
        and row["base_answer_id"] == row["donor_foil_id"]
        and row["base_foil_id"] == row["donor_answer_id"]
        for row in rows
    )


def test_every_source_is_exact_hash_bound() -> None:
    assert set(candidate.load_sources()) == set(candidate.EXPECTED_SOURCE_SHA256)
    original = candidate.EXPECTED_SOURCE_SHA256["donors"]
    candidate.EXPECTED_SOURCE_SHA256["donors"] = "0" * 64
    try:
        with pytest.raises(candidate.CrossSyntaxAuthorityError, match="immutable donors"):
            candidate.load_sources()
    finally:
        candidate.EXPECTED_SOURCE_SHA256["donors"] = original


def test_compiled_plan_is_only_two_sites_and_eight_maximum_forwards() -> None:
    plan = candidate.compile_plan()
    assert plan["site_ids"] == ["attn:11", "attn:11:head:03"]
    assert len(plan["calls"]) == 8
    assert plan["price"] == {
        "forward_calls": 8,
        "example_evaluations": 256,
        "backward_calls": 0,
        "model_updates": 0,
        "raw_numeric_evidence_bytes": 2048,
    }
    assert plan["call_manifest_sha256"] == \
        "09fee2b696c0766037e1547f5f0f7d407f3ae228dc16a0ed6c092f2795ee391a"
    assert plan["compiled_sha256"] == \
        "c79adb1b4fd3c5f3c6b9dac5cd57bedb27ebcd584d972490bede9219f9911e9a"
    assert plan["validation_scope"] == \
        "new_cross_syntax_relations_not_unseen_text"
    assert "within-construction" in plan["correction"]
    assert plan["model_loaded"] is False
    assert plan["gpu_accessed"] is False
    assert plan["queue_touched"] is False


class PassingBackend:
    """A deterministic executor with 60% recovery at both declared sites."""

    def __init__(self) -> None:
        self.native_calls = 0
        self.patched_calls = 0

    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput:
        self.native_calls += 1
        logits = tuple((3.0, 1.0) for _row_id in batch.row_ids)
        captured = {}
        if capture:
            captured = {
                (row_id, site_id): (site_id, row_id)
                for row_id in batch.row_ids
                for site_id in candidate.SITE_IDS
            }
        return producer.BatchOutput(logits, captured)

    def patched(
        self,
        batch: producer.ModelBatch,
        *,
        site,
        donor_cache,
    ) -> producer.BatchOutput:
        self.patched_calls += 1
        assert all((row_id, site.site_id) in donor_cache for row_id in batch.row_ids)
        # Native target and donor margins are both +2.  A patched target margin
        # of -0.4 gives (2 - -0.4) / (2 + 2) = 0.6 recovery.
        return producer.BatchOutput(
            tuple((0.6, 1.0) for _row_id in batch.row_ids), {}
        )


def test_fake_execution_scores_exact_row_recovery_without_gpu() -> None:
    backend = PassingBackend()
    now = datetime(2026, 9, 4, 15, 40, tzinfo=timezone.utc)
    wall_values = iter((now, now + timedelta(seconds=2)))
    monotonic_values = iter((10.0, 12.0))
    result = run.run_science(
        backend=backend,
        wall_clock=lambda: next(wall_values),
        monotonic_clock=lambda: next(monotonic_values),
    )
    assert backend.native_calls == 4
    assert backend.patched_calls == 4
    assert result["terminal"] == "screen"
    assert result["predictions"] == {
        "pred_a_native_capability": True,
        "pred_b_attention11_cross_syntax": True,
        "pred_c_head11_3_cross_syntax": True,
    }
    assert result["active_price"] == result["maximum_price"]
    assert len(result["native_evidence"]) == 128
    assert [item["site_id"] for item in result["site_results"]] == list(
        candidate.SITE_IDS
    )
    for site in result["site_results"]:
        assert site["row_count"] == 64
        assert site["overall_direction_fraction"] == 1.0
        assert site["overall_mean_recovery"] == pytest.approx(0.6)
        assert all(cell["row_count"] == 16 and cell["passed"] for cell in site["cells"])
        assert all(record["recovery"] == pytest.approx(0.6)
                   for record in site["evidence"])


def test_mutated_derived_relation_is_rejected() -> None:
    rows = candidate.build_rows()
    rows[0] = dict(rows[0], donor_subject_number=rows[0]["base_subject_number"])
    with pytest.raises(candidate.CrossSyntaxAuthorityError, match="does not change"):
        candidate.validate_rows(rows)


def test_cli_dry_run_cannot_fall_through_to_model_execution(
    monkeypatch, capsys,
) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run"])
    monkeypatch.delenv("BQLIB_DRYRUN", raising=False)
    monkeypatch.delenv("BQLIB_NO_MODEL", raising=False)
    run.cli()
    printed = json.loads(capsys.readouterr().out)
    assert printed["gpu_accessed"] is False
    assert printed["model_loaded"] is False
    assert printed["authority_sha256"] == run.EXPECTED_AUTHORITY_SHA256


def test_cli_rejects_unknown_arguments(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "--not-a-real-flag"])
    with pytest.raises(run.CrossSyntaxRunError, match="unknown command-line"):
        run.cli()
