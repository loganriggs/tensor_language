from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import run_mlp2_rank512_refit_v1 as assay
import prepare_mlp2_rank512_refit_v1_rows as rows


def test_rank512_price_and_gauge_function_replay() -> None:
    generator = torch.Generator().manual_seed(7)
    model = assay.RankBilinear(
        torch.randn(assay.RANK, assay.WIDTH, generator=generator) * 0.01,
        torch.randn(assay.RANK, assay.WIDTH, generator=generator) * 0.01,
        torch.randn(assay.WIDTH, assay.RANK, generator=generator) * 0.01,
        torch.randn(assay.WIDTH, generator=generator) * 0.01,
    )
    receipt = assay.canonicalize_minimum_norm(model)
    assert model.price()["stored_scalar_values"] == 1_770_624
    assert model.price()["products"] == 512
    assert receipt["canary_max_abs_error"] < 1e-5


def test_document_reduction_native_identity() -> None:
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(2, 256, 13, generator=generator)
    targets = torch.randint(0, 13, (2, 256), generator=generator)
    reduced = assay.reduce_document(logits, logits.clone(), targets)
    assert reduced.shape == (2, 9)
    torch.testing.assert_close(reduced[:, 1], reduced[:, 0])
    torch.testing.assert_close(reduced[:, 2], torch.zeros(2, dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(reduced[:, 3], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(reduced[:, 5], torch.full((2,), 192.0, dtype=torch.float64))


def test_row_split_is_role_disjoint() -> None:
    tensor = torch.arange(rows.TOTAL_DOCUMENTS * rows.TOKEN_LENGTH).reshape(
        rows.TOTAL_DOCUMENTS, rows.TOKEN_LENGTH,
    )
    records = [{"document_id": str(i), "dataset_document_index": i}
               for i in range(rows.TOTAL_DOCUMENTS)]
    split = rows.split_rows(tensor, records)
    assert split["TRAIN"][0].shape == (192, 257)
    assert split["EVALUATION"][0].shape == (192, 257)
    assert set(split["TRAIN"][1][0].values()).isdisjoint(
        set(split["EVALUATION"][1][0].values())
    )


def test_checkpoint_retains_tiny_literal_improvement_without_resetting_patience() -> None:
    # A 0.05% improvement is the new literal minimum and must be retained, but it is
    # below the preregistered 0.1% threshold for resetting early-stop patience.
    assert assay.checkpoint_decision(0.9995, 1.0, 1.0) == (True, False)
    assert assay.checkpoint_decision(0.998, 0.9995, 1.0) == (True, True)
    with pytest.raises(ValueError):
        assay.checkpoint_decision(float("nan"), 1.0, 1.0)


def test_centered_cancellation_is_invariant_to_input_translation() -> None:
    generator = torch.Generator().manual_seed(11)
    model = assay.RankBilinear(
        torch.randn(assay.RANK, assay.WIDTH, generator=generator) * 1e-3,
        torch.randn(assay.RANK, assay.WIDTH, generator=generator) * 1e-3,
        torch.randn(assay.WIDTH, assay.RANK, generator=generator) * 1e-3,
        torch.randn(assay.WIDTH, generator=generator),
    )
    # Repeating the entire population changes neither its global means nor ratio.
    state = torch.randn(19, assay.WIDTH, generator=generator)
    first = assay.cancellation_ratio(model, state)
    second = assay.cancellation_ratio(model, state.repeat(2, 1))
    assert first == pytest.approx(second, rel=2e-5)


def test_lock_replacement_is_detected(tmp_path: Path) -> None:
    lock = tmp_path / "claim.lock"
    claim = rows.acquire_claim(lock)
    try:
        replacement = tmp_path / "replacement"
        replacement.write_text(claim.nonce + "\n")
        os.replace(replacement, lock)
        with pytest.raises(RuntimeError, match="replaced"):
            rows.require_claim(claim, lock)
    finally:
        rows.release_claim(claim, lock)


def test_two_role_cache_verifier_rejects_hash_or_tensor_drift(tmp_path: Path) -> None:
    value = torch.arange(rows.DOCUMENTS_PER_ROLE * rows.TOKEN_LENGTH).reshape(
        rows.DOCUMENTS_PER_ROLE, rows.TOKEN_LENGTH,
    )
    path = tmp_path / "train.pt"
    torch.save(value, path)
    entry = {"file_sha256": rows.file_sha256(path),
             "tensor_sha256": rows.tensor_sha256(value)}
    torch.testing.assert_close(rows.verify_cache(path, entry), value)
    entry["file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bytes changed"):
        rows.verify_cache(path, entry)


def test_receipt_and_failure_publication_is_create_only(tmp_path: Path) -> None:
    terminal = tmp_path / "terminal.json"
    rows.write_json_create_only(terminal, {"kind": "receipt"})
    with pytest.raises(FileExistsError):
        rows.write_json_create_only(terminal, {"kind": "failure"})
    assert json.loads(terminal.read_text()) == {"kind": "receipt"}


def test_bootstrap_reducer_rejects_nonfinite_or_wrong_shape() -> None:
    ledgers = {arm: torch.ones(192, 9, dtype=torch.float64) for arm in assay.ARMS}
    ledgers["FULL512"][0, 1] = float("nan")
    with pytest.raises(RuntimeError, match="finiteness"):
        assay.bootstrap_improvements(ledgers)


def test_source_closure_contains_direct_numerical_contract_and_tests() -> None:
    names = {path.name for path in rows.SOURCE_PATHS}
    assert {
        "bilin18_observed_model_facade.py",
        "test_bilin18_observed_model_facade.py",
        "mlp2_cmr_v1_physical_program.py",
        "test_mlp2_cmr_v1_physical_program.py",
        "tt_model.py",
    } <= names
