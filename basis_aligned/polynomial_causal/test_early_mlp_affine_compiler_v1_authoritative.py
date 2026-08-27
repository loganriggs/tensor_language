from __future__ import annotations

import json

import pytest
import torch

import early_mlp_affine_compiler_v1_authoritative as runner


def test_document_block_permutation_is_whole_block_and_deterministic() -> None:
    documents = ["a", "a", "b", "b", "c", "d", "e", "f"]
    first = runner.document_block_permutation(documents, 17)
    second = runner.document_block_permutation(documents, 17)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(len(documents)))
    # The two two-row documents move as aligned two-row blocks.
    assert abs(int(first[0]) - int(first[1])) == 1
    assert abs(int(first[2]) - int(first[3])) == 1
    expanded = runner.expand_capture_permutation(first)
    assert expanded.shape == (len(documents) * 64,)
    assert sorted(expanded.tolist()) == list(range(len(documents) * 64))


def test_document_block_permutation_rejects_degenerate_single_document() -> None:
    with pytest.raises(RuntimeError, match="degenerate"):
        runner.document_block_permutation(["same", "same"], 1)


def test_mean_state_is_constant_and_has_no_cached_examples() -> None:
    target = torch.randn(30, 64)
    state = runner.mean_state(target)
    assert state["rank"] == 0
    assert state["bias"].shape == (64,)
    assert state["left"].shape == (1152, 8)
    assert state["right"].shape == (8, 64)
    assert not any("row" in key or "label" in key for key in state)


def test_gauge_variant_preserves_physical_map_and_price() -> None:
    generator = torch.Generator().manual_seed(4)
    basis = torch.linalg.qr(
        torch.randn(1152, 64, generator=generator), mode="reduced"
    ).Q
    state = {
        "mean": torch.zeros(1152), "scale": torch.ones(1152),
        "bias": torch.randn(64, generator=generator),
        "left": torch.randn(1152, 8, generator=generator),
        "right": torch.randn(8, 64, generator=generator),
        "rank": 8, "lambda": 0.001,
    }
    programs = {"main": {0: state, 1: state}}
    moved, moved_bases, diagnostics = runner.gauge_variant(
        programs, {0: basis, 1: basis}
    )
    assert set(moved) == {"main"}
    assert set(moved_bases) == {0, 1}
    assert diagnostics["0"]["physical_max_abs_error"] <= 3e-5
    assert diagnostics["0"]["price_before"] == diagnostics["0"]["price_after"]


def test_prereg_and_row_receipt_are_bound_to_runner_constants() -> None:
    prereg = json.loads(runner.PREREG.read_text())
    rows = json.loads(runner.ROWS_RECEIPT.read_text())
    assert runner.file_sha256(runner.PREREG) == runner.PREREG_SHA256
    assert runner.file_sha256(runner.ROWS_RECEIPT) == runner.ROWS_RECEIPT_SHA256
    assert prereg["experiment"] == "early_mlp_affine_compiler_v1"
    assert rows["authorized_for_training"] is True
    assert rows["training_license_sites"] == [0, 1]


def test_artifact_validator_rejects_missing_program(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "bad.pt"
    receipt = tmp_path / "receipt.json"
    torch.save({
        "status": "frozen_before_final_scoring",
        "preregistration_sha256": runner.PREREG_SHA256,
        "programs": {"main": {}},
    }, artifact)
    receipt.write_text(json.dumps({
        "status": "frozen_before_final_scoring",
        "artifact_sha256": runner.file_sha256(artifact),
    }))
    monkeypatch.setattr(runner, "ARTIFACT_RECEIPT", receipt)
    with pytest.raises(RuntimeError, match="program set"):
        runner.validate_artifact(artifact)
