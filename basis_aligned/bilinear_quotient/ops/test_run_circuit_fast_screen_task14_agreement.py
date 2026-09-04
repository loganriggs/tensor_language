#!/usr/bin/env python3
"""CPU-only tests for the thin managed Task 14 screen wrapper."""

from __future__ import annotations

import json

import circuit_experiment_spec as framework
import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_spec as screen
import circuit_prior_art
import run_circuit_fast_screen_task14_agreement as run
import run_circuit_fast_screen_task14_agreement_v2 as run_v2


def test_wrapper_binds_reviewed_authority_prior_art_and_same_answer_c() -> None:
    rows = candidate.build_rows(candidate.TASK_ID)
    assert candidate.validate_rows(rows) == run.EXPECTED_AUTHORITY_SHA256
    prior = json.loads(run.PRIOR_ART.read_text())
    assert circuit_prior_art.validate_source_files(prior, run.ROOT) == \
        run.EXPECTED_PRIOR_ART_SHA256
    spec = run.build_spec(rows)
    compiled = screen.compile_screen(spec, rows)
    assert compiled["score_contract"]["family_roles"]["C"] == \
        "same_answer_active_negative_control"
    assert compiled["price"]["forward_calls"] == 228
    assert compiled["max_price"] == {
        "phase": "FIT",
        "forward_calls": 264,
        "example_evaluations": 8448,
        "backward_calls": 0,
        "model_updates": 0,
        "evidence_bytes": 67584,
    }


def test_wrapper_and_candidate_compile_the_identical_scientific_spec() -> None:
    rows = candidate.build_rows(candidate.TASK_ID)
    wrapped = screen.spec_json(run.build_spec(rows))
    direct = screen.spec_json(candidate.build_spec(rows))
    assert wrapped == direct
    assert framework.canonical_sha256(wrapped) == \
        "3d0f91ad27d8ad75fd1250ea158958fde27dd24e6b3284a9437a4b1d53cdcf77"


def test_dryrun_is_queue_free_and_model_free() -> None:
    rows = candidate.build_rows(candidate.TASK_ID)
    receipt = screen.compile_dryrun(run.build_spec(rows), rows)
    assert receipt["call_manifest_sha256"] == \
        "02dd64b91a819da5f2b9838aafac15ebf68c52db66584f5d9cbdd3c4348cd306"
    assert receipt["compiled_sha256"] == \
        "08d5699bb986be1079c67b36ef4e4504ba0aedc754b85d6f61d79ab056f9dcd6"
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False
    assert receipt["queue_touched"] is False


def test_managed_dryrun_validates_hashes_without_creating_result(monkeypatch) -> None:
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    before = run.RESULT.read_bytes() if run.RESULT.exists() else None
    receipt = run.managed.run_managed(run.CONFIG, candidate, root=run.ROOT)
    assert receipt["authority_sha256"] == run.EXPECTED_AUTHORITY_SHA256
    assert receipt["prior_art_sha256"] == run.EXPECTED_PRIOR_ART_SHA256
    assert receipt["dryrun"]["model_loaded"] is False
    after = run.RESULT.read_bytes() if run.RESULT.exists() else None
    assert after == before


def test_v2_is_only_an_execution_fix_with_a_fresh_create_only_result() -> None:
    assert run_v2.CONFIG.expected_authority_sha256 == run.EXPECTED_AUTHORITY_SHA256
    assert run_v2.CONFIG.expected_prior_art_sha256 == run.EXPECTED_PRIOR_ART_SHA256
    assert run_v2.CONFIG.max_price == run.CONFIG.max_price
    assert run_v2.CONFIG.request_id.endswith("-v2")
    assert run_v2.CONFIG.experiment_id.endswith("-v2")
    assert run_v2.RESULT != run.RESULT
