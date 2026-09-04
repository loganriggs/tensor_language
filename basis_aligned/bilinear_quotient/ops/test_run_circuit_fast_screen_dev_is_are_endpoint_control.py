"""CPU tests for the excluded unrelated-is/are development probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import re

import pytest

from circuit_fast_screen_producer import BatchOutput
import run_circuit_fast_screen_dev_is_are_endpoint_control as probe


def test_bank_is_balanced_counterfactual_and_one_batch() -> None:
    batch, dryrun = probe.compile_probe()
    assert dryrun["bank_sha256"] == probe.EXPECTED_BANK_SHA256
    assert dryrun["scientific_status"] == "excluded_development_only"
    assert dryrun["split"] == "DEVELOPMENT"
    assert dryrun["reuse_policy"] == "forbidden_in_FIT_SELECT_TEST_OOD"
    assert dryrun["price"] == {
        "forward_calls": 1, "example_evaluations": 32,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 256,
    }
    assert len(batch.row_ids) == len(probe.EXAMPLES) == 32
    assert set(batch.answer_ids) == {318, 389}
    assert set(batch.foil_ids) == {318, 389}

    by_pair: dict[str, list[probe.ProbeExample]] = defaultdict(list)
    by_remap: dict[str, list[probe.ProbeExample]] = defaultdict(list)
    for example in probe.EXAMPLES:
        by_pair[example.counterfactual_pair_id].append(example)
        by_remap[example.remap_pair_id].append(example)
        assert len(re.findall(r"\bis\b", example.prompt.lower())) == 1
        assert len(re.findall(r"\bare\b", example.prompt.lower())) == 1
    assert len(by_pair) == 16
    for rows in by_pair.values():
        assert len(rows) == 2
        assert {row.truth_value for row in rows} == {"yes", "no"}
        assert {row.answer for row in rows} == {" is", " are"}
        assert len({(row.family, row.mapping_id, row.label_order) for row in rows}) == 1
    assert len(by_remap) == 16
    for rows in by_remap.values():
        assert len(rows) == 2
        assert {row.mapping_id for row in rows} == {"yes_is", "yes_are"}
        assert {row.answer for row in rows} == {" is", " are"}
        assert len({
            (row.family, row.label_order, row.truth_value, row.claim_text)
            for row in rows
        }) == 1

    for feature in ("family", "mapping_id", "label_order", "truth_value", "claim_text"):
        answers: dict[str, Counter[str]] = defaultdict(Counter)
        for example in probe.EXAMPLES:
            answers[getattr(example, feature)][example.answer] += 1
        assert all(set(counts) == {" is", " are"} for counts in answers.values())


def test_dryrun_never_loads_model_or_writes_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.Bilin18TorchBackend, "load",
        lambda *_args: pytest.fail("dryrun loaded the model"),
    )
    receipt = probe.run_probe(
        root=tmp_path, environment={"BQLIB_DRYRUN": "1"},
    )
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False
    assert receipt["balance"]["response_remap_pairs"] == 16
    assert receipt["causal_followup_semantics"]["identification_limit"].startswith(
        "these rows are development-only"
    )
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()


def test_cli_supports_explicit_dry_run_and_rejects_unknown_arguments() -> None:
    assert probe.parse_args(["--dry-run"]).dry_run is True
    with pytest.raises(SystemExit):
        probe.parse_args(["--unknown-argument"])


def test_no_model_alias_is_also_pure_dryrun(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.Bilin18TorchBackend, "load",
        lambda *_args: pytest.fail("no-model dryrun loaded the model"),
    )
    receipt = probe.run_probe(root=tmp_path, environment={"BQLIB_NO_MODEL": "1"})
    assert receipt["price"]["forward_calls"] == 1
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()


def test_fake_native_call_writes_create_only_capable_receipt(tmp_path) -> None:
    class FakeBackend:
        calls = 0

        def native(self, batch, *, capture):
            self.calls += 1
            assert capture is False
            return BatchOutput(
                answer_foil=tuple((2.0, -1.0) for _ in batch.row_ids),
                captured={},
            )

    backend = FakeBackend()
    moments = iter((
        datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc) + timedelta(seconds=1.25),
    ))
    ticks = iter((4.0, 5.25))
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
        "serial_seconds": 1.25, "forward_calls": 1,
        "example_evaluations": 32, "backward_calls": 0,
        "model_updates": 0, "evidence_bytes": 256,
    }
    assert result["capability"]["development_wording_capable"] is True
    assert result["capability"]["global_accuracy"] == 1.0
    assert result["capability"]["endpoint_accuracy"] == {" are": 1.0, " is": 1.0}
    assert result["capability"]["fully_correct_counterfactual_pair_count"] == 16
    assert len(result["capability"]["family_mapping_cells"]) == 8
    assert all(cell["passed"] for cell in result["capability"]["family_mapping_cells"])
    assert result["predictions"]["pred_d_wording_capable"] is True
    assert len(result["evidence"]) == 32
    assert summary["development_wording_capable"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        probe.run_probe(root=tmp_path, environment={}, backend=backend)
    assert backend.calls == 1


def test_capability_rule_rejects_one_bad_family_mapping_cell() -> None:
    evidence = []
    for example in probe.EXAMPLES:
        correct = not (
            example.family == "arithmetic"
            and example.mapping_id == "yes_is"
            and example.truth_value == "yes"
        )
        evidence.append({**probe._example_json(example), "correct": correct})
    capability = probe._capability_summary(evidence)
    assert capability["development_wording_capable"] is False
    failed = [
        cell for cell in capability["family_mapping_cells"]
        if cell["family"] == "arithmetic" and cell["mapping_id"] == "yes_is"
    ]
    assert len(failed) == 1
    assert failed[0]["accuracy"] == 0.5
    assert failed[0]["passed"] is False


def test_bank_mutation_fails_frozen_hash() -> None:
    changed = list(probe.EXAMPLES)
    changed[0] = replace(changed[0], prompt=changed[0].prompt.replace("twelve", "12"))
    with pytest.raises(probe.EndpointControlError, match="frozen digest"):
        probe.compile_probe(changed)


def test_nonfinite_evidence_fails_without_receipt(tmp_path) -> None:
    class NonfiniteBackend:
        def native(self, batch, *, capture):
            pairs = [(1.0, 0.0) for _ in batch.row_ids]
            pairs[11] = (float("nan"), 0.0)
            return BatchOutput(tuple(pairs), {})

    with pytest.raises(probe.EndpointControlError, match="nonfinite"):
        probe.run_probe(root=tmp_path, environment={}, backend=NonfiniteBackend())
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()
