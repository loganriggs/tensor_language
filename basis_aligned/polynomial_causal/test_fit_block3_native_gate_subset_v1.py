import math
import json

import pytest
import torch

import fit_block3_native_gate_subset_v1 as fitter


def test_training_sse_matches_direct_residual():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(120, 9, generator=generator, dtype=torch.float64)
    y = torch.randn(120, 5, generator=generator, dtype=torch.float64)
    gram = x.T @ x
    cross = x.T @ y
    decoder = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    observed = fitter.training_sse(gram, cross, decoder, y.square().sum())
    expected = float((x @ decoder.T - y).square().sum())
    assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-10)


def test_fit_constants_match_frozen_protocol():
    assert fitter.BUDGETS == (256, 512)
    assert fitter.OMP_BATCH == 16
    assert fitter.RIDGE == 1e-6
    assert fitter.RANDOM_SEED == 2026082907


def test_fit_one_materializes_executable_float32_program_and_literal_price():
    generator = torch.Generator().manual_seed(9)
    width, gates, selected = 4, 7, 3
    left = torch.randn(gates, width, generator=generator, dtype=torch.float32)
    right = torch.randn(gates, width, generator=generator, dtype=torch.float32)
    bias = torch.randn(width, generator=generator, dtype=torch.float32)
    x = torch.randn(40, gates, generator=generator, dtype=torch.float64)
    y = torch.randn(40, width, generator=generator, dtype=torch.float64)
    gram, cross = x.T @ x, x.T @ y
    indices = torch.tensor([1, 3, 5])
    program, metrics = fitter._fit_one(
        left=left, right=right, bias=bias,
        prefilter_indices=torch.arange(gates), gram=gram,
        real_cross=cross, fit_cross=cross, local_indices=indices,
        write_energy=y.square().sum(),
    )
    assert program.left.dtype == program.right.dtype == program.decoder.dtype == torch.float32
    assert program.write(torch.zeros(2, width, dtype=torch.float32)).shape == (2, width)
    assert metrics["gate_count"] == selected
    assert metrics["float_byte_count"] == 4 * metrics["float_parameter_count"]
    assert metrics["total_literal_byte_count"] == (
        metrics["float_byte_count"] + metrics["index_byte_count"]
    )
    assert metrics["product_count_per_token"] == selected
    assert metrics["deployed_polarization_replay_relative"] <= 2e-5


def _configure_tiny_transaction(monkeypatch, tmp_path, *, fail_terminal=False):
    for name in ("FIT_AUTHORITY", "PROGRAMS", "RESULTS", "RECEIPT", "FAILURE", "LOCK"):
        monkeypatch.setattr(fitter, name, tmp_path / name.lower())
    collector_receipt_path = tmp_path / "collector_receipt.json"
    collector_receipt_path.write_text("{}\n")
    monkeypatch.setattr(fitter.collector, "RECEIPT", collector_receipt_path)
    monkeypatch.setattr(fitter.collector, "WIDTH", 2)
    monkeypatch.setattr(fitter.collector, "GATES", 5)
    monkeypatch.setattr(fitter.collector, "PREFILTER", 3)
    monkeypatch.setattr(fitter, "BUDGETS", (1, 2))
    monkeypatch.setattr(fitter, "OMP_BATCH", 1)
    checkpoint = fitter.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="a" * 64,
        weights_sha256="b" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=10,
    )
    authority = {"authority_sha256": "c" * 64}
    collector_receipt = {"checkpoint_weights_sha256": checkpoint.weights_sha256}
    payload = {
        "prefilter_indices": torch.tensor([0, 1, 2]),
        "prefilter_gram": torch.eye(3, dtype=torch.float64),
        "prefilter_cross": torch.zeros(3, 2, dtype=torch.float64),
        "prefilter_permuted_cross": torch.zeros(3, 2, dtype=torch.float64),
        "contribution_energy": torch.tensor([3., 2., 1., 0., 0.], dtype=torch.float64),
        "native_typed_write_energy": torch.tensor(100., dtype=torch.float64),
    }
    input_hashes = {"collector": "d" * 64}
    monkeypatch.setattr(
        fitter, "_validate_payload",
        lambda: (authority, collector_receipt, payload, input_hashes),
    )
    source = {"commit": "e" * 40, "sha256s": {}, "sha256": "f" * 64}
    monkeypatch.setattr(fitter, "_source_closure", lambda: source)
    monkeypatch.setattr(fitter, "_verify_source_closure", lambda value: None)
    checks = 0

    def verify_inputs(value):
        nonlocal checks
        checks += 1
        if fail_terminal and checks == 3:
            raise RuntimeError("injected terminal collector-input drift")

    monkeypatch.setattr(fitter, "_verify_collector_inputs", verify_inputs)
    monkeypatch.setattr(fitter.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    prefix = f"transformer.h.{fitter.collector.LAYER}.mlp."
    generator = torch.Generator().manual_seed(11)
    state = {
        prefix + "Left.weight": torch.randn(5, 2, generator=generator),
        prefix + "Right.weight": torch.randn(5, 2, generator=generator),
        prefix + "Down_bias": torch.randn(2, generator=generator),
    }
    monkeypatch.setattr(fitter.torch, "load", lambda *args, **kwargs: state)
    return checkpoint


def test_fit_transaction_publishes_float32_programs_and_receipt_last(monkeypatch, tmp_path):
    _configure_tiny_transaction(monkeypatch, tmp_path)
    result = fitter.run()
    assert fitter.FIT_AUTHORITY.exists() and fitter.PROGRAMS.exists() and fitter.RESULTS.exists()
    assert fitter.RECEIPT.exists() and not fitter.FAILURE.exists() and not fitter.LOCK.exists()
    receipt = json.loads(fitter.RECEIPT.read_text())
    assert receipt["status"] == "deterministic_fit_complete_no_evaluation_opened"
    assert result["native_baseline_price"]["dtype"] == "torch.float32"
    assert all(arm["deployed_dtype"] == "torch.float32" for arm in result["arms"].values())


def test_fit_terminal_drift_publishes_failure_without_receipt(monkeypatch, tmp_path):
    _configure_tiny_transaction(monkeypatch, tmp_path, fail_terminal=True)
    with pytest.raises(RuntimeError, match="injected terminal"):
        fitter.run()
    assert fitter.FIT_AUTHORITY.exists() and fitter.PROGRAMS.exists() and fitter.RESULTS.exists()
    assert fitter.FAILURE.exists() and not fitter.RECEIPT.exists() and not fitter.LOCK.exists()
