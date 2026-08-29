from __future__ import annotations

import io
import inspect

import pytest
import torch

import calibrate_mlp2_cmr_v1_fit_selector as calibration


def synthetic_rows() -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(
        calibration.DOCUMENTS * (calibration.SEQUENCE + 1), dtype=torch.long,
    ).remainder(101).view(calibration.DOCUMENTS, calibration.SEQUENCE + 1)
    eligible = torch.zeros(calibration.DOCUMENTS, calibration.SEQUENCE, dtype=torch.bool)
    eligible[:, calibration.SCORE_START:] = True
    valid = torch.nonzero(eligible.flatten(), as_tuple=False).flatten()
    eligible.zero_()
    eligible.view(-1)[valid[:calibration.ELIGIBLE_POSITIONS]] = True
    assert int(eligible.sum()) == calibration.ELIGIBLE_POSITIONS
    return rows, eligible


def test_frequency_reference_counts_only_eligible_next_token_targets() -> None:
    rows, eligible = synthetic_rows()
    counts, bins = calibration.target_frequency_reference(rows, eligible)
    expected = torch.bincount(
        rows[:, 1:][eligible], minlength=calibration.VOCAB,
    ).long()
    assert torch.equal(counts, expected)
    assert int(counts.sum()) == calibration.ELIGIBLE_POSITIONS
    assert int(bins.sum()) == calibration.ELIGIBLE_POSITIONS
    assert tuple(bins.shape) == (9,)


def test_copy_cells_partition_eligible_and_use_nearest_successor() -> None:
    rows, eligible = synthetic_rows()
    rows.zero_()
    eligible.zero_()
    # At p=64, nearest previous token 7 is j=60 and its successor at 61 is 9.
    rows[0, 60], rows[0, 61], rows[0, 64], rows[0, 65] = 7, 9, 7, 9
    # At p=66 the nearest repeat exists, but its successor differs from target.
    rows[0, 66], rows[0, 67] = 7, 8
    # At p=68 token 5 has no previous occurrence.
    rows[0, 68], rows[0, 69] = 5, 4
    eligible[0, 64] = eligible[0, 66] = eligible[0, 68] = True
    cells = calibration.nearest_repeat_cells(rows, eligible)
    assert bool(cells["copy_positive"][0, 64])
    assert bool(cells["repeat_negative"][0, 66])
    assert bool(cells["nonrepeat"][0, 68])
    assert sum(int(value.sum()) for value in cells.values()) == 3


def test_epsilon_grid_is_fit_margin_only_sorted_positive_and_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "ELIGIBLE_POSITIONS", 1000)
    margins = torch.linspace(0.0, 10.0, 1000, dtype=torch.float64)
    quantiles, grid = calibration.epsilon_grid(margins)
    quantiles2, grid2 = calibration.epsilon_grid(margins.clone())
    assert torch.equal(quantiles, quantiles2)
    assert torch.equal(grid, grid2)
    assert torch.all(grid > 0) and torch.all(grid[1:] > grid[:-1])
    assert set(2.0 ** exponent for exponent in calibration.DYADIC_EXPONENTS) <= set(
        grid.tolist()
    )


def test_margin_grid_rejects_wrong_count_negative_or_nonfinite(monkeypatch) -> None:
    monkeypatch.setattr(calibration, "ELIGIBLE_POSITIONS", 3)
    with pytest.raises(ValueError, match="malformed"):
        calibration.epsilon_grid(torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="malformed"):
        calibration.epsilon_grid(torch.tensor([1.0, float("nan"), 2.0], dtype=torch.float64))


def test_source_closure_and_authority_boundaries_are_explicit() -> None:
    closure = {path.name for path in calibration.SOURCE_CLOSURE}
    assert {
        "MLP2_CMR_V1_PREREGISTRATION.md",
        "MLP2_CMR_V1_MARGIN_FREQUENCY_ADDENDUM.md",
        "COPY_SOURCE_EDGE_DISCOVERY_PREREGISTRATION.md",
        "calibrate_mlp2_cmr_v1_fit_selector.py",
        "test_calibrate_mlp2_cmr_v1_fit_selector.py",
        "project_mlp2_cmr_v1_fit_selector_rows.py",
        "test_project_mlp2_cmr_v1_fit_selector_rows.py",
        "bilin18_observed_model_facade.py",
        "tt_model.py",
    } <= closure
    assert not hasattr(calibration, "collect")
    source = inspect.getsource(calibration._collect)
    assert "projection.validate_role(role)" in source
    assert "ROLE_ROWS_SHA256" in source
    assert "VALIDATION" not in source
    assert "REPLICATION" not in source
    assert "CPU fallback is forbidden" in source
    assert 'torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090"' in source
    assert "capability.consumed = True" in source
    assert "attention_calls_by_site" in source
    assert "mlp_calls_by_site" in source
    main_source = inspect.getsource(calibration.main)
    success_branch = main_source[:main_source.index("    except BaseException")]
    assert success_branch.rfind("write_create_only(RECEIPT") > success_branch.rfind(
        "write_create_only(RESULT"
    )
    receipt_end = success_branch.rfind("write_create_only(RECEIPT")
    assert "write_" not in success_branch[receipt_end + len("write_create_only(RECEIPT"):]
    assert success_branch.count("final_guard(") >= 2
    assert 'summary, bundle = _collect(parent_bytes["role_rows"], capability)' in success_branch


def test_protected_inputs_parses_the_same_captured_receipt_bytes() -> None:
    original = calibration.SUFFIX_RECEIPT.read_bytes()
    # The implementation must use the captured mapping rather than a second
    # SUFFIX_RECEIPT.read_text call after hashing.
    source = inspect.getsource(calibration.protected_inputs)
    assert 'captured = {name: path.read_bytes()' in source
    assert 'json.loads(captured["suffix_receipt"])' in source
    assert 'json.loads(captured["role_manifest"])' in source
    assert 'json.loads(captured["role_receipt"])' in source
    assert 'json.loads(captured["correction_receipt"])' in source
    assert "SUFFIX_RECEIPT.read_text" not in source
    assert original


def test_role_only_artifact_has_exact_fit_semantics_and_no_role_names() -> None:
    raw = calibration.ROLE_ROWS.read_bytes()
    assert calibration.file_sha256(calibration.ROLE_ROWS) == calibration.ROLE_ROWS_SHA256
    role = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    summary = calibration.projection.validate_role(role)
    assert set(role) == {
        "document_indices", "rows", "eligible_mask", "original_token_counts",
        "clipped_token_counts",
    }
    assert summary["documents"] == 192
    assert summary["eligible_positions"] == 31_505
    assert summary["support_documents"] == 191
    assert summary["all_false_ordinals"] == [82]


def test_token_only_bundle_replay_recomputes_frequency_and_copy_cells() -> None:
    raw = calibration.ROLE_ROWS.read_bytes()
    role = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    frequency, _ = calibration.target_frequency_reference(
        role["rows"], role["eligible_mask"],
    )
    cells = calibration.nearest_repeat_cells(role["rows"], role["eligible_mask"])
    margins = torch.linspace(
        0.0, 16.0, calibration.ELIGIBLE_POSITIONS, dtype=torch.float64,
    )
    quantiles, grid = calibration.epsilon_grid(margins)
    bundle = {
        "schema": "mlp2_cmr_v1_fit_selector_calibration_bundle",
        "fit_token_counts": frequency,
        "frequency_boundaries": torch.tensor(
            calibration.FREQUENCY_BOUNDARIES, dtype=torch.long,
        ),
        "margin_quantiles": quantiles,
        "epsilon_grid": grid,
        "fit_copy_cells": cells,
    }
    calibration.validate_bundle_replay(bundle, bundle, raw)


def test_certificate_norm_is_frozen_as_vocabulary_sum() -> None:
    addendum = calibration.ADDENDUM.read_text()
    assert r"D_2=\frac1N\sum_{i=1}^N\sum_{v=1}^{50304}" in addendum
    assert "vocabulary **sum**, not a mean per logit" in addendum


def test_call_and_frequency_contract_is_frozen() -> None:
    assert calibration.DOCUMENTS == 192
    assert calibration.BATCH == 4
    assert calibration.CALLS == 48
    assert calibration.ELIGIBLE_POSITIONS == 31_505
    assert calibration.FREQUENCY_BOUNDARIES == (1, 2, 4, 8, 16, 32, 64, 128)
    assert calibration.MARGIN_QUANTILES[0] == 0.001
    assert calibration.MARGIN_QUANTILES[-1] == 0.999
