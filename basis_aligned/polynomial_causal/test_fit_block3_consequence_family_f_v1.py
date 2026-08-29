import ast
import json
from pathlib import Path

import pytest
import torch

import block3_consequence_family_f_call_ledger as call_contract
import fit_block3_consequence_family_f_v1 as runner
import native_gate_subset as subset


def test_runner_source_closure_and_permissions_are_fit_only():
    assert str(runner.RUNNER.relative_to(runner.ROOT)) in runner.SOURCE_PATHS
    assert "basis_aligned/polynomial_causal/test_fit_block3_consequence_family_f_v1.py" in (
        runner.SOURCE_PATHS
    )
    assert "basis_aligned/polynomial_causal/block3_consequence_family_f_call_ledger.py" in (
        runner.SOURCE_PATHS
    )
    source = runner.RUNNER.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("validate_block3" in name or "final" in name for name in imports)
    checkpoint = runner.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="a" * 64,
        weights_sha256="b" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=10,
    )
    authority = runner.authority(
        {"sha256": "c" * 64}, {"sha256": "d" * 64},
        {"sha256": "e" * 64}, checkpoint,
    )
    assert authority["authorized_for_fit_execution"] is True
    assert authority["authorized_for_validation"] is False
    assert authority["authorized_for_final"] is False
    assert authority["authorized_for_global_ledger_credit"] is False


def test_rows_refuse_to_deserialize_before_authority(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "AUTHORITY", tmp_path / "missing_authority.json")
    loaded = []
    monkeypatch.setattr(
        runner.torch, "load",
        lambda *args, **kwargs: loaded.append(args) or torch.empty(0),
    )
    with pytest.raises(RuntimeError, match="before authority"):
        runner.load_rows_after_authority({})
    assert loaded == []


def test_affine_write_preserves_double_parameter_gradients_with_float32_program():
    generator = torch.Generator().manual_seed(0)
    program = subset.NativeGateSubsetProgram(
        indices=torch.arange(3),
        left=torch.randn(3, 4, generator=generator),
        right=torch.randn(3, 4, generator=generator),
        decoder=torch.randn(4, 3, generator=generator),
        bias=torch.randn(4, generator=generator),
    )
    z = torch.randn(2, 5, 4, generator=generator)
    scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
    correction = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    write = runner._affine_write(program, z, scale, correction)
    assert write.dtype == torch.float32
    write.square().sum().backward()
    assert scale.grad is not None and scale.grad.dtype == torch.float64
    assert correction.grad is not None and correction.grad.dtype == torch.float64


def test_build_programs_has_nested_real_support_and_same_support_control(monkeypatch):
    monkeypatch.setattr(runner, "PREFILTER", 6)
    monkeypatch.setattr(runner, "BUDGETS", (2, 4))
    generator = torch.Generator().manual_seed(1)
    width, gates = 3, 8
    left = torch.randn(gates, width, generator=generator)
    right = torch.randn(gates, width, generator=generator)
    down = torch.randn(width, gates, generator=generator)
    bias = torch.randn(width, generator=generator)
    prefilter = torch.tensor([0, 2, 3, 4, 6, 7])
    x = torch.randn(30, 6, generator=generator, dtype=torch.float64)
    y = torch.randn(30, width, generator=generator, dtype=torch.float64)
    gram, cross = x.T @ x, x.T @ y
    scores = {
        arm: torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float64)
        for arm in call_contract.SCORE_ARMS
    }
    programs, supports = runner.build_programs(
        left=left, right=right, native_down=down, native_bias=bias,
        prefilter_indices=prefilter, gram=gram, cross=cross,
        permuted_cross=cross.flip(0), scores=scores,
    )
    assert set(supports["teacher_k2"].tolist()) < set(supports["teacher_k4"].tolist())
    for budget in (2, 4):
        assert torch.equal(
            programs[f"real_F_post_refit_k{budget}"].indices,
            programs[f"same_support_permuted_cross_post_refit_k{budget}"].indices,
        )
        assert programs[f"real_F_binary_native_down_k{budget}"].gates == budget


def _valid_program_artifact(monkeypatch):
    monkeypatch.setattr(runner, "WIDTH", 2)
    scores = {
        arm: torch.cat((torch.ones(512), torch.zeros(512))).double()
        for arm in call_contract.SCORE_ARMS
    }
    prefilter = torch.arange(1024)
    supports = {
        f"{arm}_k{budget}": prefilter[:budget].clone()
        for arm in call_contract.SCORE_ARMS for budget in (256, 512)
    }
    random_order = torch.randperm(
        1024, generator=torch.Generator().manual_seed(runner.RANDOM_SEED),
    )
    supports.update({
        f"random_k{budget}": prefilter[random_order[:budget]].clone()
        for budget in (256, 512)
    })
    programs = {}
    for name in set(call_contract.REPORT_STUDENT_ARMS) - {"continuous_teacher_F1"}:
        gates = 256 if "k256" in name else 512
        budget = gates
        if name.startswith("random_post_refit"):
            indices = supports[f"random_k{budget}"].clone()
        elif name.startswith("row_reversal_selector"):
            indices = supports[f"teacher_row_reversal_k{budget}"].clone()
        elif name.startswith("document_derangement_selector"):
            indices = supports[f"teacher_document_derangement_k{budget}"].clone()
        else:
            indices = supports.get(f"teacher_k{budget}", torch.arange(gates)).clone()
        programs[name] = {
            "indices": indices,
            "left": torch.ones(gates, 2),
            "right": torch.ones(gates, 2),
            "decoder": torch.ones(2, gates),
            "bias": torch.zeros(2),
        }
    promotive = ["real_F_post_refit_k256", "real_F_post_refit_k512"]
    artifact = {
        "schema": "block3_consequence_family_f_v1_programs",
        "authority_sha256": "a" * 64,
        "scores": scores,
        "supports": supports,
        "programs": programs,
        "affine_parameters": {
            arm: {
                "scale_float64": 1.0,
                "correction_float64": torch.zeros(2, dtype=torch.float64),
            }
            for arm in call_contract.AFFINE_ARMS
        },
        "promotive_programs": promotive,
        "nonpromotive_programs": sorted(set(programs) - set(promotive)),
    }
    monkeypatch.setattr(
        runner.torch, "load", lambda *args, **kwargs: {"prefilter_indices": prefilter},
    )
    return artifact


def test_semantic_program_reload_rejects_control_support_drift(monkeypatch):
    artifact = _valid_program_artifact(monkeypatch)
    runner.semantic_validate_program_artifact(artifact)
    artifact["programs"]["same_support_permuted_cross_post_refit_k256"][
        "indices"
    ] = torch.arange(1, 257)
    with pytest.raises(RuntimeError, match="support provenance"):
        runner.semantic_validate_program_artifact(artifact)


def _configure_transaction(monkeypatch, tmp_path, *, terminal_drift=False):
    for name in ("AUTHORITY", "PROGRAMS", "RESULTS", "RECEIPT", "FAILURE", "LOCK"):
        monkeypatch.setattr(runner, name, tmp_path / name.lower())
    monkeypatch.setattr(runner.life, "require_pristine_namespace", lambda: None)
    source = {"commit": "a" * 40, "paths": {}, "sha256": "b" * 64}
    prior = {"sha256": "c" * 64}
    rows = {"sha256": "d" * 64}
    checkpoint = runner.facade.CheckpointReceipt(
        revision="r", snapshot="s", config_sha256="e" * 64,
        weights_sha256="f" * 64, weights_bytes=1,
        tokenizer_vocab=10, logit_vocab=10,
    )
    monkeypatch.setattr(runner, "source_closure", lambda: source)
    monkeypatch.setattr(runner.life, "prior_artifact_binding", lambda: prior)
    monkeypatch.setattr(runner.life, "row_binding", lambda: rows)
    monkeypatch.setattr(runner.facade, "validate_snapshot", lambda **kwargs: checkpoint)
    checks = []

    def verify(*args):
        checks.append(True)
        if terminal_drift and len(checks) == 4:
            raise RuntimeError("injected terminal source drift")

    monkeypatch.setattr(runner, "verify_frozen_inputs", verify)
    monkeypatch.setattr(runner, "semantic_validate_program_artifact", lambda value: None)

    def execute_fit(**kwargs):
        assert runner.AUTHORITY.exists()
        call_contract.record_frozen_schedule(kwargs["calls"])
        receipt = kwargs["calls"].validate_exact()
        return ({"fake_tensor": torch.arange(3)}, {
            "schema": "fake_result", "call_ledger": receipt,
            "maximum_allocated_cuda_bytes": 0,
        })

    monkeypatch.setattr(runner, "execute_fit", execute_fit)
    return checks


def test_transaction_is_authority_first_and_receipt_last(monkeypatch, tmp_path):
    checks = _configure_transaction(monkeypatch, tmp_path)
    result = runner.run()
    assert len(checks) == 4
    assert result["schema"] == "fake_result"
    assert runner.AUTHORITY.exists() and runner.PROGRAMS.exists() and runner.RESULTS.exists()
    assert runner.RECEIPT.exists() and not runner.FAILURE.exists() and not runner.LOCK.exists()
    receipt = json.loads(runner.RECEIPT.read_text())
    assert receipt["status"] == "fit_complete_receipt_last_no_evaluation_opened"
    assert receipt["validation_rows_loaded"] == receipt["final_rows_loaded"] == 0


def test_terminal_drift_preserves_partial_outputs_without_receipt(monkeypatch, tmp_path):
    _configure_transaction(monkeypatch, tmp_path, terminal_drift=True)
    with pytest.raises(RuntimeError, match="injected terminal"):
        runner.run()
    assert runner.AUTHORITY.exists() and runner.PROGRAMS.exists() and runner.RESULTS.exists()
    assert runner.FAILURE.exists() and not runner.RECEIPT.exists() and not runner.LOCK.exists()
    failure = json.loads(runner.FAILURE.read_text())
    assert failure["partial_call_ledger"]["complete"] is True
