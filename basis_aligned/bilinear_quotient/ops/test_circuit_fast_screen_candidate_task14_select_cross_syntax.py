#!/usr/bin/env python3
# BQLANE: cpu

import circuit_fast_screen_candidate_task14_select_cross_syntax as candidate
import circuit_battery_task14 as task14
import pytest


def test_select_rows_are_balanced_and_frozen() -> None:
    rows = candidate.build_rows()
    assert candidate.validate_rows(rows) == \
        "ecaae3b5e7baddcc3e9d7b888133ad78f8f6185656bbb439a044248bd58157c1"
    assert len(rows) == 64
    cells = {name: sum(row["cell_id"] == name for row in rows)
             for name in {row["cell_id"] for row in rows}}
    assert set(cells.values()) == {16}
    assert all(row["split"] == "SELECT" for row in rows)


def test_select_nouns_and_templates_are_disjoint_from_fit() -> None:
    authority, _ = task14.build_authority()
    fit, _ = task14.split_rows(authority, "FIT")
    select, _ = task14.split_rows(authority, "SELECT")
    fit_nouns = {form for row in fit for field in (
        "head_pair", "attractor_pair", "second_head_pair", "surface_attractor_pair",
        "second_attractor_pair"
    ) for form in row[field]}
    select_nouns = {form for row in select for field in (
        "head_pair", "attractor_pair", "second_head_pair", "surface_attractor_pair",
        "second_attractor_pair"
    ) for form in row[field]}
    assert fit_nouns.isdisjoint(select_nouns)
    fit_templates = {row[field] for row in fit
                     for field in ("base_template_id", "donor_template_id")}
    select_templates = {row[field] for row in select
                        for field in ("base_template_id", "donor_template_id")}
    assert fit_templates.isdisjoint(select_templates)


def test_plan_is_targeted_and_cpu_only() -> None:
    plan = candidate.compile_plan()
    assert plan["compiled_sha256"] == \
        "c9c84cfa23ac7081d5af8a5848ef4e97634f5a937d16f52bedc2eb5bbb326ef4"
    assert plan["site_ids"] == ["attn:11", "attn:11:head:03"]
    assert plan["price"] == {
        "forward_calls": 8, "example_evaluations": 256,
        "backward_calls": 0, "model_updates": 0,
        "raw_numeric_evidence_bytes": 2048,
    }


def test_self_consistent_but_noncanonical_row_is_rejected() -> None:
    rows = candidate.build_rows()
    rows[0] = dict(rows[0], base_text=rows[0]["base_text"] + " altered")
    with pytest.raises(
        candidate.SelectCrossSyntaxAuthorityError,
        match="exact regenerated SELECT authority",
    ):
        candidate.validate_rows(rows)
