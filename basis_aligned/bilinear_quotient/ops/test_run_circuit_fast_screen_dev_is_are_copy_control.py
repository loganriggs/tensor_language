"""CPU tests for the excluded natural is/are copy-capability probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import re

import pytest

from circuit_fast_screen_producer import BatchOutput
import run_circuit_fast_screen_dev_is_are_copy_control as probe


def test_bank_is_balanced_and_pairs_change_only_selection() -> None:
    batch, dryrun = probe.compile_probe()
    assert dryrun["bank_sha256"] == probe.EXPECTED_BANK_SHA256
    assert dryrun["scientific_status"] == "excluded_development_only"
    assert dryrun["split"] == "DEVELOPMENT"
    assert dryrun["reuse_policy"] == "forbidden_in_FIT_SELECT_TEST_OOD"
    assert dryrun["minimum_family_direction_accuracy"] == 0.75
    assert dryrun["price"] == {
        "forward_calls": 1,
        "example_evaluations": 32,
        "backward_calls": 0,
        "model_updates": 0,
        "evidence_bytes": 256,
    }
    assert len(batch.row_ids) == len(probe.EXAMPLES) == 32
    assert set(batch.answer_ids) == {318, 389}
    assert set(batch.foil_ids) == {318, 389}

    pairs: dict[str, list[probe.ProbeExample]] = defaultdict(list)
    for example in probe.EXAMPLES:
        pairs[example.selection_pair_id].append(example)
        assert len(re.findall(r"\bis\b", example.prompt.lower())) == 1
        assert len(re.findall(r"\bare\b", example.prompt.lower())) == 1
    assert len(pairs) == 16
    for rows in pairs.values():
        assert {row.answer for row in rows} == {" is", " are"}
        assert {row.selected_endpoint for row in rows} == {"is", "are"}
        assert len({probe._selection_frame(row.prompt, row.selected_key) for row in rows}) == 1
        assert len({(row.family, row.instance_id, row.key_order) for row in rows}) == 1

    per_cell = Counter((row.family, row.selected_endpoint) for row in probe.EXAMPLES)
    assert len(per_cell) == 8
    assert set(per_cell.values()) == {4}
    answers_by_order: dict[str, set[str]] = defaultdict(set)
    for row in probe.EXAMPLES:
        answers_by_order[row.key_order].add(row.answer)
    assert all(answers == {" is", " are"} for answers in answers_by_order.values())


def test_explicit_dry_run_never_loads_model_or_writes(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(
        probe.Bilin18TorchBackend,
        "load",
        lambda *_args: pytest.fail("--dry-run loaded the model"),
    )
    probe.main(["--dry-run"])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_loaded"] is False
    assert receipt["gpu_accessed"] is False
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()


def test_unknown_cli_argument_is_rejected_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe,
        "run_probe",
        lambda **_kwargs: pytest.fail("unknown argument reached execution"),
    )
    with pytest.raises(SystemExit) as caught:
        probe.main(["--unknown-option"])
    assert caught.value.code == 2


def test_environment_dryrun_is_still_supported(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe.Bilin18TorchBackend,
        "load",
        lambda *_args: pytest.fail("environment dryrun loaded the model"),
    )
    receipt = probe.run_probe(
        root=tmp_path,
        environment={"BQLIB_DRYRUN": "1", "BQLIB_NO_MODEL": "1"},
    )
    assert receipt["price"]["forward_calls"] == 1
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()


def test_fake_native_call_writes_create_only_capable_receipt(tmp_path) -> None:
    class FakeBackend:
        calls = 0

        def native(self, batch, *, capture):
            self.calls += 1
            assert capture is False
            return BatchOutput(
                answer_foil=tuple((3.0, -1.0) for _ in batch.row_ids),
                captured={},
            )

    backend = FakeBackend()
    moments = iter((
        datetime(2026, 9, 4, 16, 30, tzinfo=timezone.utc),
        datetime(2026, 9, 4, 16, 30, tzinfo=timezone.utc)
        + timedelta(seconds=0.75),
    ))
    ticks = iter((8.0, 8.75))
    summary = probe.run_probe(
        root=tmp_path,
        environment={},
        backend=backend,
        wall_clock=lambda: next(moments),
        timer=lambda: next(ticks),
    )
    assert backend.calls == 1
    path = tmp_path / probe.RESULT_RELATIVE
    result = json.loads(path.read_text())
    assert result["scientific_status"] == (
        "excluded_development_only_not_screen_evidence"
    )
    assert result["runtime"] == {
        "serial_seconds": 0.75,
        "forward_calls": 1,
        "example_evaluations": 32,
        "backward_calls": 0,
        "model_updates": 0,
        "evidence_bytes": 256,
    }
    assert result["capability"]["development_copy_capable"] is True
    assert result["capability"]["global_accuracy"] == 1.0
    assert len(result["capability"]["family_direction_cells"]) == 8
    assert all(cell["accuracy"] == 1.0 for cell in result["capability"]["family_direction_cells"])
    assert result["predictions"]["pred_d_native_copy_capable"] is True
    assert len(result["evidence"]) == 32
    assert summary["development_copy_capable"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        probe.run_probe(root=tmp_path, environment={}, backend=backend)
    assert backend.calls == 1


def test_capability_requires_75_percent_in_each_family_direction() -> None:
    evidence = []
    failed_rows = {
        "color_card.red_blue.is_first.select_is",
        "color_card.red_blue.are_first.select_is",
    }
    for example in probe.EXAMPLES:
        evidence.append({
            **probe._example_json(example),
            "correct": example.row_id not in failed_rows,
        })
    capability = probe._capability_summary(evidence)
    assert capability["development_copy_capable"] is False
    cell = next(
        item for item in capability["family_direction_cells"]
        if item["family"] == "color_card" and item["selected_endpoint"] == "is"
    )
    assert cell["example_count"] == 4
    assert cell["accuracy"] == 0.5
    assert cell["passed"] is False


def test_bank_mutation_fails_frozen_hash() -> None:
    changed = list(probe.EXAMPLES)
    changed[0] = replace(changed[0], prompt=changed[0].prompt + " ")
    with pytest.raises(probe.CopyControlError, match="paired prompts|frozen digest"):
        probe.compile_probe(changed)


def test_nonfinite_output_fails_without_receipt(tmp_path) -> None:
    class NonfiniteBackend:
        def native(self, batch, *, capture):
            pairs = [(1.0, 0.0) for _ in batch.row_ids]
            pairs[5] = (float("inf"), 0.0)
            return BatchOutput(tuple(pairs), {})

    with pytest.raises(probe.CopyControlError, match="nonfinite"):
        probe.run_probe(root=tmp_path, environment={}, backend=NonfiniteBackend())
    assert not (tmp_path / probe.RESULT_RELATIVE).exists()
