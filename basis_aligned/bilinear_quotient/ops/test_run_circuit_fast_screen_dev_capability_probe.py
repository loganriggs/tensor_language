"""CPU tests for the scientifically excluded native wording probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

import circuit_fast_screen_candidate_pronoun as pronoun_bank
import circuit_fast_screen_candidate_quote_parity as quote_bank
from circuit_fast_screen_producer import BatchOutput
import run_circuit_fast_screen_dev_capability_probe as probe


def test_frozen_development_bank_is_balanced_disjoint_and_one_batch() -> None:
    batch, dryrun = probe.compile_probe()
    assert dryrun["bank_sha256"] == probe.EXPECTED_BANK_SHA256
    assert dryrun["scientific_status"] == "excluded_development_only"
    assert dryrun["split"] == "DEVELOPMENT"
    assert dryrun["reuse_policy"] == "forbidden_in_FIT_SELECT_TEST_OOD"
    assert dryrun["behavior_count"] == 3
    assert dryrun["template_count"] == 11
    assert dryrun["example_count"] == len(batch.row_ids) == 44
    assert dryrun["price"] == {
        "forward_calls": 1, "example_evaluations": 44,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 352,
    }
    templates: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for example in probe.EXAMPLES:
        templates[(example.behavior, example.template_id)][example.answer] += 1
    assert len(templates) == 11
    assert all(sorted(counts.values()) == [2, 2] for counts in templates.values())
    frozen_prompts = {
        row[side]
        for bank in (
            quote_bank.build_rows(quote_bank.TASK_ID),
            pronoun_bank.build_rows(pronoun_bank.TASK_ID),
        )
        for row in bank
        for side in ("base_text", "donor_text")
    }
    assert not ({example.prompt for example in probe.EXAMPLES} & frozen_prompts)


def test_dryrun_never_loads_backend_or_creates_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.Bilin18TorchBackend, "load",
        lambda *_: pytest.fail("dryrun loaded model backend"),
    )
    receipt = probe.run_probe(
        root=tmp_path, environment={"BQLIB_DRYRUN": "1"},
    )
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False
    assert receipt["registered_predictions"] == dict(probe.REGISTERED_PREDICTIONS)
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()


def test_no_model_alias_is_also_a_pure_dryrun(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.Bilin18TorchBackend, "load",
        lambda *_: pytest.fail("no-model dryrun loaded backend"),
    )
    receipt = probe.run_probe(root=tmp_path, environment={"BQLIB_NO_MODEL": "1"})
    assert receipt["price"]["forward_calls"] == 1
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()


def test_fake_native_call_writes_exact_create_only_development_receipt(tmp_path) -> None:
    class FakeBackend:
        calls = 0

        def native(self, batch, *, capture):
            self.calls += 1
            assert capture is False
            return BatchOutput(
                answer_foil=tuple((2.5, -0.5) for _ in batch.row_ids),
                captured={},
            )

    backend = FakeBackend()
    moments = iter((
        datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=1.5),
    ))
    ticks = iter((10.0, 11.5))
    summary = probe.run_probe(
        root=tmp_path,
        environment={},
        backend=backend,
        wall_clock=lambda: next(moments),
        timer=lambda: next(ticks),
    )
    assert backend.calls == 1
    result_path = tmp_path / probe.RESULT_RELATIVE
    result = json.loads(result_path.read_text())
    assert result["scientific_status"] == (
        "excluded_development_only_not_screen_evidence"
    )
    assert result["reuse_policy"] == "forbidden_in_FIT_SELECT_TEST_OOD"
    assert result["runtime"] == {
        "serial_seconds": 1.5, "forward_calls": 1,
        "example_evaluations": 44, "backward_calls": 0,
        "model_updates": 0, "evidence_bytes": 352,
    }
    assert result["predictions"] == {
        "pred_a_runtime_one_native_batch": True,
        "pred_b_coverage_complete": True,
        "pred_c_integrity_finite_aligned": True,
    }
    assert len(result["template_answer_accuracy"]) == 22
    assert {cell["accuracy"] for cell in result["template_answer_accuracy"]} == {1.0}
    assert len(result["evidence"]) == 44
    assert {item["answer_minus_foil_margin"] for item in result["evidence"]} == {3.0}
    assert summary["result_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        probe.run_probe(root=tmp_path, environment={}, backend=backend)
    assert backend.calls == 1


def test_coherent_wording_mutation_fails_frozen_bank_hash() -> None:
    changed = list(probe.EXAMPLES)
    changed[0] = replace(changed[0], prompt=changed[0].prompt + " today")
    with pytest.raises(probe.DevelopmentProbeError, match="frozen digest"):
        probe.compile_probe(changed)


def test_nonfinite_backend_evidence_fails_without_receipt(tmp_path) -> None:
    class NonfiniteBackend:
        def native(self, batch, *, capture):
            pairs = [(1.0, 0.0) for _ in batch.row_ids]
            pairs[7] = (float("nan"), 0.0)
            return BatchOutput(tuple(pairs), {})

    with pytest.raises(probe.DevelopmentProbeError, match="nonfinite"):
        probe.run_probe(root=tmp_path, environment={}, backend=NonfiniteBackend())
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()
