from __future__ import annotations

import math

import pytest
import torch

import run_block3_native_down_behavioral_port_v1 as port


def test_parent_hashes_and_receipt_join_are_exact():
    assert port.verify_parent_files() == port.PARENT_PINS


def test_random_and_shift_controls_are_deterministic_and_distinct():
    first, second = port.random_support(), port.random_support()
    assert torch.equal(first, second)
    port.validate_support(first)
    assert torch.equal(port.shifted_decoder_indices(first), torch.roll(first, -1))
    assert not torch.equal(first, port.shifted_decoder_indices(first))


def test_fit_error_pca_known_answer_orientation_scale_and_hash(monkeypatch):
    monkeypatch.setattr(port, "WIDTH", 5)
    generator = torch.Generator().manual_seed(4)
    scales = torch.tensor([5., 4., 3., 2., 1.], dtype=torch.float64)
    errors = torch.randn(400, 5, generator=generator, dtype=torch.float64) * scales
    directions, receipt = port.fit_error_directions(errors)
    assert directions.shape == (4, 5)
    assert torch.allclose(directions.square().mean(1), torch.ones(4, dtype=torch.float64), atol=1e-12)
    for direction in directions:
        coordinate = int(torch.argmax(direction.abs()))
        assert direction[coordinate] > 0
    assert receipt.fit_error_rms == pytest.approx(float(errors.square().mean().sqrt()))
    assert len(receipt.directions_sha256) == 64


def test_fit_error_pca_rejects_unresolved_tie(monkeypatch):
    monkeypatch.setattr(port, "WIDTH", 5)
    errors = torch.cat((torch.eye(5, dtype=torch.float64), -torch.eye(5, dtype=torch.float64)))
    with pytest.raises(RuntimeError, match="eigengap"):
        port.fit_error_directions(errors)


def test_cell_census_and_exact_call_plan():
    assert len(port.error_secant_cells()) == 4
    assert len(port.edit_cells()) == 16
    assert port.expected_call_plan() == {
        "fit_prefixes": 120,
        "fresh_prefixes": 48,
        "ordinary_full_model_replays": 48,
        "teacher_suffixes": 48,
        "student_suffixes_per_batch": 40,
        "student_suffixes": 1920,
        "all_suffixes": 1968,
        "physical_calls_sites_0_3_per_kind": 216,
        "physical_calls_sites_4_17_per_kind": 2016,
        "native_mlp3_calls": 216,
        "compiled_student_native_mlp3_calls": 0,
        "error_candidate_plus_one_reused": True,
        "optimizer_calls": 0,
        "backward_calls": 0,
    }


def test_document_bootstrap_is_seeded_and_ratio_uses_document_sums(monkeypatch):
    monkeypatch.setattr(port, "BOOTSTRAP_DRAWS", 8)
    first, second = port.bootstrap_weights(3), port.bootstrap_weights(3)
    assert torch.equal(first, second)
    numerator = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    denominator = torch.tensor([[2., 2., 2.]], dtype=torch.float64)
    series = port.ratio_series(numerator, denominator, first)
    assert series.shape == (9, 1)
    assert series[0, 0] == 1.0


def test_simultaneous_bounds_use_max_deviation(monkeypatch):
    monkeypatch.setattr(port, "BOOTSTRAP_DRAWS", 4)
    series = torch.tensor([
        [1., 10.], [2., 10.], [1., 12.], [0., 9.], [1.5, 10.5],
    ], dtype=torch.float64)
    result = port.simultaneous_bounds(series)
    upper_radius = float(torch.quantile(torch.tensor([1., 2., -1., .5], dtype=torch.float64), .95))
    assert result["simultaneous_q95_upper"][0] == pytest.approx(1 + upper_radius)
    assert result["simultaneous_q95_upper"][1] == pytest.approx(10 + upper_radius)


def test_centered_logits_remove_only_per_position_constant():
    value = torch.tensor([[[1., 2., 3.], [4., 4., 4.]]])
    centered = port.centered_logits(value)
    assert torch.allclose(centered.mean(-1), torch.zeros(1, 2))
    assert torch.equal(centered[0, 0], torch.tensor([-1., 0., 1.]))


def test_source_closure_contains_all_direct_contract_sources():
    for path in (
        port.PREREG, port.ADDENDUM, port.RUNNER, port.TEST,
        port.ROW_FREEZER, port.ROW_FREEZER_TEST,
    ):
        assert str(path.relative_to(port.ROOT)) in port.SOURCE_PATHS


def test_run_is_fail_closed_before_authority(monkeypatch):
    monkeypatch.setattr(port, "verify_parent_files", lambda: port.PARENT_PINS)
    monkeypatch.setattr(port, "source_closure", lambda: {"commit": "x"})
    with pytest.raises(RuntimeError, match="remains NO-GO"):
        port.run()
    assert not any(path.exists() for path in port.output_namespace())
