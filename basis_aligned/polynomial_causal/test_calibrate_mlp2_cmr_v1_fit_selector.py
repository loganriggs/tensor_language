from __future__ import annotations

import copy
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
    assert "_consume_capability(capability)" in source
    assert "attention_calls_by_site" in source
    assert "mlp_calls_by_site" in source
    main_source = inspect.getsource(calibration.main)
    success_branch = main_source[:main_source.index("    except BaseException")]
    assert success_branch.rfind("RECEIPT, canonical_json_bytes(receipt)") > (
        success_branch.rfind("RESULT, canonical_json_bytes(summary)")
    )
    assert "before_link=receipt_prelink_guard" in success_branch
    assert "before_link=failure_prelink_guard" in main_source
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
    frequency, frequency_bins = calibration.target_frequency_reference(
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
    }
    role_summary = calibration.projection.validate_role(role)
    checkpoint = {"weights_sha256": calibration.facade.WEIGHTS_SHA256}
    result = {
        "schema": "mlp2_cmr_v1_fit_selector_calibration_result",
        "status": "fit_selector_calibration_complete_no_validation_or_replication",
        "checkpoint": checkpoint,
        "checkpoint_after_load": checkpoint,
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "model_dtype": str(torch.bfloat16),
        "strict_state_dict_load": True,
        "role_summary": role_summary,
        "documents": calibration.DOCUMENTS,
        "eligible_positions": calibration.ELIGIBLE_POSITIONS,
        "forward_calls": calibration.CALLS,
        "forward_returns": calibration.CALLS,
        "backward_calls": 0,
        "attention_calls": 18 * calibration.CALLS,
        "mlp_calls": 18 * calibration.CALLS,
        "attention_calls_by_site": [calibration.CALLS] * 18,
        "mlp_calls_by_site": [calibration.CALLS] * 18,
        "margin_quantiles": {
            format(q, ".6g"): float(value)
            for q, value in zip(calibration.MARGIN_QUANTILES, quantiles.tolist())
        },
        "margin_minimum": 0.0,
        "margin_mean": 8.0,
        "margin_maximum": 16.0,
        "epsilon_grid": grid.tolist(),
        "epsilon_grid_count": int(grid.numel()),
        "frequency_boundaries": list(calibration.FREQUENCY_BOUNDARIES),
        "fit_frequency_bin_counts": frequency_bins.tolist(),
        "copy_cell_counts": {name: int(mask.sum()) for name, mask in cells.items()},
        "tensor_hashes": {
            "fit_token_counts": calibration.tensor_sha256(frequency),
            "margin_quantiles": calibration.tensor_sha256(quantiles),
            "epsilon_grid": calibration.tensor_sha256(grid),
            **{
                f"fit_{name}_mask": calibration.tensor_sha256(mask)
                for name, mask in cells.items()
            },
        },
        "runtime_seconds": 1.0,
        "validation_opened": False,
        "replication_opened": False,
        "finite_candidate_constructed": False,
        "raw_logits_published": False,
    }
    calibration.validate_output_semantics(bundle, result, raw)
    corrupted = copy.deepcopy(result)
    corrupted["forward_calls"] = calibration.CALLS - 1
    with pytest.raises(RuntimeError, match="call-ledger"):
        calibration.validate_output_semantics(bundle, corrupted, raw)
    corrupted_bundle = copy.deepcopy(bundle)
    corrupted_bundle["fit_token_counts"][0] += 1
    with pytest.raises(RuntimeError, match="token-only"):
        calibration.validate_output_semantics(corrupted_bundle, result, raw)
    extra_grid_bundle = copy.deepcopy(bundle)
    extra_grid_bundle["epsilon_grid"] = torch.tensor(
        sorted(set(grid.tolist()) | {0.123456789}), dtype=torch.float64,
    )
    extra_grid_result = copy.deepcopy(result)
    extra_grid_result["epsilon_grid"] = extra_grid_bundle["epsilon_grid"].tolist()
    extra_grid_result["epsilon_grid_count"] = int(
        extra_grid_bundle["epsilon_grid"].numel()
    )
    extra_grid_result["tensor_hashes"]["epsilon_grid"] = calibration.tensor_sha256(
        extra_grid_bundle["epsilon_grid"],
    )
    with pytest.raises(RuntimeError, match="exact frozen union"):
        calibration.validate_output_semantics(
            extra_grid_bundle, extra_grid_result, raw,
        )


def test_capability_is_single_mint_nonconstructible_noncopyable(tmp_path, monkeypatch) -> None:
    lock = tmp_path / "claim.json"
    authority_path = tmp_path / "authority.json"
    nonce = "capability-test-nonce"
    lock.write_bytes(calibration.canonical_json_bytes({"nonce": nonce}))
    inode_stat = lock.stat(follow_symlinks=False)
    inode = (inode_stat.st_dev, inode_stat.st_ino)
    authority = {
        "status": "authority_frozen_before_calibration_model_access",
        "authorized_role": "FIT_SELECTOR",
        "authorized_forward_calls": calibration.CALLS,
        "authorized_backward_calls": 0,
    }
    authority_path.write_bytes(calibration.canonical_json_bytes(authority))
    monkeypatch.setattr(calibration, "LOCK", lock)
    monkeypatch.setattr(calibration, "AUTHORITY", authority_path)
    authority_hash = calibration.file_sha256(authority_path)
    with pytest.raises(TypeError, match="not directly constructible"):
        calibration._CalibrationCapability(object(), nonce, inode, authority_hash)
    capability = calibration._mint_capability(nonce, inode, authority_hash)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(capability)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(capability)
    with pytest.raises(RuntimeError, match="already minted"):
        calibration._mint_capability(nonce, inode, authority_hash)
    calibration._consume_capability(capability)
    assert capability.consumed
    with pytest.raises(RuntimeError, match="fresh"):
        calibration._consume_capability(capability)


@pytest.mark.parametrize("target_name,opposite_name", [
    ("receipt.json", "failure.json"), ("failure.json", "receipt.json"),
])
def test_terminal_writer_blocks_a_late_opposite_artifact(
    tmp_path, target_name: str, opposite_name: str,
) -> None:
    target = tmp_path / target_name
    opposite = tmp_path / opposite_name

    def inject_late_opposite() -> None:
        opposite.write_bytes(b"late rival")

    with pytest.raises(RuntimeError, match="terminal namespace"):
        calibration.write_create_only(
            target, b"candidate terminal",
            before_link=lambda: calibration.terminal_prelink_guard(
                target, opposite, inject_late_opposite,
            ),
        )
    assert opposite.read_bytes() == b"late rival"
    assert not target.exists()


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
