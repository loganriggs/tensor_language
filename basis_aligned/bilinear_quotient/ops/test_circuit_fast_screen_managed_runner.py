"""Focused CPU tests for the generic managed screen and thin wrappers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

import circuit_fast_screen_candidates as sentence_candidate
import circuit_fast_screen_candidate_pronoun as pronoun_candidate
import circuit_fast_screen_ledger as ledger
import circuit_fast_screen_managed_runner as managed
import circuit_fast_screen_producer as producer
import run_circuit_fast_screen_pronoun as pronoun
import run_circuit_fast_screen_quote_parity as quote
import run_circuit_fast_screen_sentence_terminal as sentence


def test_sentence_wrapper_preserves_reviewed_semantics_and_dryrun() -> None:
    rows = sentence_candidate.build_rows(sentence_candidate.TASK_ID)
    spec = sentence.build_spec(rows)
    assert spec.hypothesis.behavior == "sentence_terminal.semantic_choice"
    assert spec.hypothesis.information_read == (
        "whether the unfinished sentence is declarative or interrogative"
    )
    assert spec.hypothesis.proposed_operation == (
        "carry that sentence-mode state across two syntactic constructions"
    )
    assert spec.semantic_position.role == (
        "final input token before the predicted punctuation"
    )
    assert managed.framework.canonical_sha256(managed.screen.spec_json(spec)) == (
        "fe16317e39aa0adcf0351fccc7af49ddfdf3347f7bafde45cedd883e744f9239"
    )
    receipt = managed.run_managed(
        sentence.CONFIG, sentence_candidate, root=sentence.ROOT,
        environment={"BQLIB_DRYRUN": "1"},
        science_runner=lambda *_: pytest.fail("dryrun invoked science"),
    )
    assert receipt["prior_art_sha256"] == sentence.EXPECTED_PRIOR_ART_SHA256
    assert receipt["authority_sha256"] == sentence.EXPECTED_AUTHORITY_SHA256
    assert receipt["execution_policy"] == "managed_queue_only"
    assert receipt["dryrun"]["active_price"]["forward_calls"] == 228
    assert receipt["dryrun"]["max_price"] == {
        "phase": "FIT", "forward_calls": 264,
        "example_evaluations": 8448, "backward_calls": 0,
        "model_updates": 0, "evidence_bytes": 67584,
    }


def test_pronoun_wrapper_compiles_current_candidate_and_exact_hypothesis() -> None:
    rows = pronoun_candidate.build_rows(pronoun_candidate.TASK_ID)
    spec = pronoun.build_spec(rows)
    assert spec.authority_sha256 == pronoun.EXPECTED_AUTHORITY_SHA256
    assert spec.hypothesis.information_read == (
        "which explicitly gendered person performed the action"
    )
    assert spec.hypothesis.proposed_operation == (
        "bind the selected event participant to the correct gendered pronoun "
        "across active and passive voice"
    )
    assert spec.hypothesis.proposed_write == "evidence for he versus she"
    assert spec.hypothesis.alternative_explanation == (
        "construction-specific actor-position cue or generic he/she output service"
    )
    assert spec.hypothesis.circuit_prediction == (
        "one site transfers A1 and A2 while sparing location P and visible-pronoun C"
    )
    assert spec.hypothesis.opposing_null_prediction == (
        "native capability fails or no site transfers both constructions selectively"
    )
    receipt = managed.run_managed(
        pronoun.CONFIG, pronoun_candidate, root=pronoun.ROOT,
        environment={"BQLIB_NO_MODEL": "1"},
        science_runner=lambda *_: pytest.fail("dryrun invoked science"),
    )
    assert receipt["prior_art_sha256"] == pronoun.EXPECTED_PRIOR_ART_SHA256
    assert receipt["dryrun"]["max_price"]["forward_calls"] == 264
    assert receipt["dryrun"]["max_price"]["example_evaluations"] == 8448
    assert receipt["dryrun"]["max_price"]["evidence_bytes"] == 67584


def test_quote_parity_wrapper_compiles_exact_hypothesis_and_dryrun() -> None:
    rows = quote.candidate.build_rows(quote.candidate.TASK_ID)
    spec = quote.build_spec(rows)
    assert spec.authority_sha256 == quote.EXPECTED_AUTHORITY_SHA256
    assert spec.hypothesis.information_read == "whether one ASCII quote remains pending"
    assert spec.hypothesis.proposed_operation == (
        "carry quote-count parity across surface constructions"
    )
    assert spec.hypothesis.proposed_write == (
        "evidence for a closing quote rather than a period"
    )
    assert spec.hypothesis.alternative_explanation == (
        "the known closer output or a generic period-versus-quote endpoint direction"
    )
    assert spec.hypothesis.circuit_prediction == (
        "one site transfers both pending-quote constructions and spares both controls"
    )
    assert spec.hypothesis.opposing_null_prediction == (
        "native capability fails or no common selective site transfers quote state"
    )
    receipt = managed.run_managed(
        quote.CONFIG, quote.candidate, root=quote.ROOT,
        environment={"BQLIB_DRYRUN": "1"},
        science_runner=lambda *_: pytest.fail("dryrun invoked science"),
    )
    assert receipt["prior_art_sha256"] == quote.EXPECTED_PRIOR_ART_SHA256
    assert receipt["execution_policy"] == "managed_queue_only"
    assert receipt["dryrun"]["active_price"]["forward_calls"] == 228
    assert receipt["dryrun"]["max_price"] == {
        "phase": "FIT", "forward_calls": 264,
        "example_evaluations": 8448, "backward_calls": 0,
        "model_updates": 0, "evidence_bytes": 67584,
    }


def test_reviewed_prior_digest_is_a_runtime_gate() -> None:
    changed = replace(sentence.CONFIG, expected_prior_art_sha256="0" * 64)
    with pytest.raises(managed.ManagedScreenError, match="prior-art receipt differs"):
        managed.run_managed(
            changed, sentence_candidate, root=sentence.ROOT,
            environment={"BQLIB_DRYRUN": "1"},
            science_runner=lambda *_: pytest.fail("digest mismatch invoked science"),
        )


def test_generic_real_path_writes_strict_result_then_exact_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        pronoun.CONFIG,
        request_id="pronoun-managed-harness-test",
        prior_art_relative="prior.json",
        result_relative="results/pronoun.json",
        ledger_relative="ledger/screens.jsonl",
    )
    monkeypatch.setattr(
        managed,
        "load_prior_art",
        lambda actual, candidate, root: ({
            "candidate_id": candidate.TASK_ID,
            "relation": "extension",
            "novelty_delta": "test-only novelty",
        }, actual.expected_prior_art_sha256),
    )
    run = producer.FastScreenRun(
        terminal="null",
        reason="no_selective_causal_site",
        selected_site=None,
        head_stage="skipped_no_parent",
        capability_cells=(),
        native_logits=(),
        intervention_logits=(),
        site_results=(),
        ranking=(),
        timing=producer.RunTiming(
            forward_calls=228, example_evaluations=7296, seconds=4.0,
        ),
    )
    times = iter((
        datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc) + timedelta(seconds=4),
    ))
    summary = managed.run_managed(
        config, pronoun_candidate, root=tmp_path, environment={},
        science_runner=lambda *_: run, clock=lambda: next(times),
    )
    result_path = tmp_path / config.result_relative
    result = json.loads(result_path.read_text())
    assert summary["terminal"] == result["terminal"] == "null"
    assert result["execution_policy"] == "managed_queue_only"
    assert "queue_only_science" not in result
    assert result["run"]["native_logits"] == []
    assert result["run"]["timing"] == {
        "example_evaluations": 7296, "forward_calls": 228, "seconds": 4.0,
    }
    entries = ledger.read_ledger(
        tmp_path / config.ledger_relative, result_root=tmp_path,
    )
    assert len(entries) == 1
    assert entries[0]["active_forward_calls"] == 228
    assert entries[0]["active_example_evaluations"] == 7296
    assert entries[0]["active_evidence_bytes"] == 58368
    assert entries[0]["max_forward_calls"] == 264
    assert entries[0]["max_example_evaluations"] == 8448
    assert entries[0]["max_evidence_bytes"] == 67584
