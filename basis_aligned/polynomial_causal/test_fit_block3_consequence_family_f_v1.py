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


def test_parent_load_detects_hash_drift_during_deserialization(monkeypatch, tmp_path):
    payload_path = tmp_path / "parent.pt"
    a_path = tmp_path / "family_a.pt"
    payload_path.write_bytes(b"parent-before")
    a_path.write_bytes(b"a-before")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.collector, "PAYLOAD", payload_path)
    prior_paths = list(runner.life.PRIOR_PATHS)
    prior_paths[4] = a_path
    monkeypatch.setattr(runner.life, "PRIOR_PATHS", tuple(prior_paths))
    authority_path = tmp_path / "authority.json"
    monkeypatch.setattr(runner, "AUTHORITY", authority_path)
    hashes = {
        "parent.pt": runner.file_sha256(payload_path),
        "family_a.pt": runner.file_sha256(a_path),
    }
    frozen = {
        "prior_artifact_binding": {
            "file_sha256s": hashes,
            "collector_authority_sha256": "c" * 64,
            "fit_authority_sha256": "f" * 64,
        }
    }
    authority_path.write_text(json.dumps(frozen))
    loads = 0

    def drifting_load(path, **kwargs):
        nonlocal loads
        loads += 1
        if loads == 1:
            payload_path.write_bytes(b"parent-after")
            return {
                "authority_sha256": "c" * 64,
                "prefilter_indices": torch.arange(runner.PREFILTER),
            }
        return {"fit_authority_sha256": "f" * 64}

    monkeypatch.setattr(runner.torch, "load", drifting_load)
    monkeypatch.setattr(runner, "require_resource_ceiling", lambda started: (0.0, 0))
    with pytest.raises(RuntimeError, match="changed during load"):
        runner.load_parent_tensors_after_authority(frozen, started=0.0)


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


def test_affine_write_is_the_exact_prospective_folded_float32_program():
    generator = torch.Generator().manual_seed(17)
    program = subset.NativeGateSubsetProgram(
        indices=torch.arange(3),
        left=torch.randn(3, 4, generator=generator),
        right=torch.randn(3, 4, generator=generator),
        decoder=torch.randn(4, 3, generator=generator),
        bias=torch.randn(4, generator=generator),
    )
    z = torch.randn(2, 5, 4, generator=generator)
    scale = torch.tensor(0.713, dtype=torch.float64, requires_grad=True)
    correction = torch.randn(4, generator=generator, dtype=torch.float64).requires_grad_()
    observed = runner._affine_write(program, z, scale, correction)
    folded = runner.core.fold_affine_calibration(
        program, scale.detach().float(), correction.detach().float(),
    )
    torch.testing.assert_close(observed.detach(), folded.write(z), rtol=0, atol=0)


def test_raw_replay_rejects_discrepancy_hidden_by_saturated_softcap(monkeypatch):
    monkeypatch.setattr(runner, "POSITION_START", 0)
    monkeypatch.setattr(runner, "POSITION_STOP", 1)
    teacher_raw = torch.tensor([[[1000.0, 0.0]]])
    ordinary_raw = torch.tensor([[[2000.0, 0.0]]])
    teacher = runner.softcap_logits(teacher_raw)
    assert float((runner.softcap_logits(ordinary_raw) - teacher).abs().max()) == 0.0
    with pytest.raises(RuntimeError, match="autonomous suffix replay"):
        runner.validate_native_replay(ordinary_raw, teacher_raw, teacher)


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
    prefilter = torch.arange(1024)
    runner.semantic_validate_program_artifact(
        artifact, expected_authority_sha256="a" * 64, prefilter=prefilter,
    )
    artifact["programs"]["same_support_permuted_cross_post_refit_k256"][
        "indices"
    ] = torch.arange(1, 257)
    with pytest.raises(RuntimeError, match="support provenance"):
        runner.semantic_validate_program_artifact(
            artifact, expected_authority_sha256="a" * 64, prefilter=prefilter,
        )


def test_semantic_program_rejects_authority_score_and_tensor_corruption(monkeypatch):
    artifact = _valid_program_artifact(monkeypatch)
    prefilter = torch.arange(1024)
    reconstructed = {
        name: runner._materialize_program(payload, device="cpu")
        for name, payload in artifact["programs"].items()
    }
    with pytest.raises(RuntimeError, match="schema"):
        runner.semantic_validate_program_artifact(
            artifact, expected_authority_sha256="b" * 64, prefilter=prefilter,
        )
    artifact = _valid_program_artifact(monkeypatch)
    artifact["scores"]["teacher"][0] = 1.1
    artifact["scores"]["teacher"][512] = -0.1
    with pytest.raises(RuntimeError, match="score vector"):
        runner.semantic_validate_program_artifact(
            artifact, expected_authority_sha256="a" * 64, prefilter=prefilter,
        )
    artifact = _valid_program_artifact(monkeypatch)
    artifact["programs"]["real_F_post_refit_k256"]["decoder"][0, 0] += 1
    with pytest.raises(RuntimeError, match="reconstructed program tensor"):
        runner.semantic_validate_program_artifact(
            artifact, expected_authority_sha256="a" * 64, prefilter=prefilter,
            reconstructed_programs=reconstructed,
        )


def _valid_result():
    ledger = call_contract.FamilyFCallLedger()
    call_contract.record_frozen_schedule(ledger)
    call_receipt = ledger.validate_exact()
    score_trace_fields = {
        "document_kl": 0.1, "row_kl": 0.1, "score_min": 0.0,
        "score_max": 1.0, "score_sum": 512.0, "saturated_zero": 0.5,
        "saturated_one": 0.5, "gradient_norm_max": 0.1,
    }
    affine_trace_fields = {
        "document_kl": 0.1, "row_kl": 0.1, "gradient_norm_max": 0.1,
        "scale": 1.0, "correction_rms": 0.0, "correction_norm": 0.0,
    }
    report_names = set(call_contract.REPORT_STUDENT_ARMS)
    program_names = report_names - {"continuous_teacher_F1"}
    overlaps = {
        f"teacher_vs_{comparison}_k{budget}": {"intersection": 1, "jaccard": 0.1}
        for comparison in (
            "teacher_row_reversal", "teacher_document_derangement", "random", "family_A",
        )
        for budget in runner.BUDGETS
    }
    price = {
        "float_values": 1, "float_bytes": 4, "index_bytes": 8, "total_bytes": 12,
        "products_per_token": 1, "linear_multiplies_per_token": 1,
    }
    transition = {
        "continuous_F1_document_kl": 0.1, "binary_native_down_document_kl": 0.1,
        "post_refit_document_kl": 0.1, "binary_minus_continuous_document_kl": 0.0,
        "refit_minus_binary_document_kl": 0.0,
    }
    return {
        "schema": "block3_consequence_family_f_v1_fit_results",
        "status": "fit_complete_no_validation_or_final_opened",
        "authority_sha256": "a" * 64,
        "score_traces": {
            arm: [{"epoch": epoch, **score_trace_fields} for epoch in range(1, 9)]
            for arm in call_contract.SCORE_ARMS
        },
        "affine_traces": {
            arm: [{"epoch": epoch, **affine_trace_fields} for epoch in range(1, 5)]
            for arm in call_contract.AFFINE_ARMS
        },
        "known_answer_replay": {
            "raw_max_absolute": 0.0, "raw_max_relative": 0.0,
            "max_absolute": 0.0, "max_relative": 0.0, "teacher_self_kl_max": 0.0,
        },
        "postfit_report": {
            arm: {
                "document_balanced_teacher_kl": 0.1, "row_mean_teacher_kl": 0.1,
                "summed_write_nrmse": 0.2,
            }
            for arm in report_names
        },
        "postfit_stage_transitions": {"256": transition, "512": transition},
        "stacked_typed_fit_nrmse": {
            arm: 0.2 for arm in program_names if not arm.startswith("affine_")
        },
        "direct_polarization_replay": {
            arm: {"max_absolute": 0.0, "max_relative": 0.0} for arm in program_names
        },
        "score_projection_replay_max_abs": {
            arm: 0.0 for arm in call_contract.SCORE_ARMS
        },
        "support_overlaps": overlaps,
        "program_prices": {arm: price for arm in program_names},
        "call_ledger": call_receipt,
        "model_state_before_sha256": "b" * 64,
        "model_state_after_sha256": "b" * 64,
        "fit_rows_loaded": runner.life.ROW_COUNT,
        "validation_rows_loaded": 0, "final_rows_loaded": 0,
        "ground_truth_target_tokens_used": 0, "retained_teacher_logits": 0,
        "authorized_for_validation": False, "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
        "elapsed_seconds": 1.0, "maximum_allocated_cuda_bytes": 0,
        "torch_version": "test", "python_version": "test",
        "programs_file_sha256": "c" * 64,
    }


def test_result_semantic_replay_rejects_missing_arm_and_bad_join():
    result = _valid_result()
    runner.semantic_validate_result(
        result, expected_authority_sha256="a" * 64,
        expected_programs_file_sha256="c" * 64,
    )
    broken = dict(result)
    broken["authority_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="joins"):
        runner.semantic_validate_result(
            broken, expected_authority_sha256="a" * 64,
            expected_programs_file_sha256="c" * 64,
        )
    impossible = _valid_result()
    impossible["postfit_report"]["continuous_teacher_F1"][
        "document_balanced_teacher_kl"
    ] = -7.0
    with pytest.raises(RuntimeError, match="impossible metrics"):
        runner.semantic_validate_result(
            impossible, expected_authority_sha256="a" * 64,
            expected_programs_file_sha256="c" * 64,
        )
    impossible_trace = _valid_result()
    impossible_trace["score_traces"]["teacher"][0]["score_sum"] = 0.0
    impossible_trace["score_traces"]["teacher"][0]["document_kl"] = -100.0
    impossible_trace["affine_traces"]["teacher_F_k512"][0][
        "correction_norm"
    ] = -3.0
    with pytest.raises(RuntimeError, match="trace contains impossible"):
        runner.semantic_validate_result(
            impossible_trace, expected_authority_sha256="a" * 64,
            expected_programs_file_sha256="c" * 64,
        )
    result["postfit_report"] = dict(result["postfit_report"])
    result["postfit_report"].pop("continuous_teacher_F1")
    with pytest.raises(RuntimeError, match="postfit report"):
        runner.semantic_validate_result(
            result, expected_authority_sha256="a" * 64,
            expected_programs_file_sha256="c" * 64,
        )


def test_result_reconstruction_rejects_self_consistent_schema_with_fake_nrmse(monkeypatch):
    artifact = _valid_program_artifact(monkeypatch)
    programs = {
        name: runner._materialize_program(payload, device="cpu")
        for name, payload in artifact["programs"].items()
    }
    parent = {
        "prefilter_indices": torch.arange(1024),
        "prefilter_gram": torch.eye(1024, dtype=torch.float64),
        "prefilter_cross": torch.zeros(1024, 2, dtype=torch.float64),
        "native_typed_write_energy": torch.tensor(1_000_000.0, dtype=torch.float64),
    }
    family_a = {256: torch.arange(256), 512: torch.arange(512)}
    result = _valid_result()
    result["program_prices"] = {
        name: runner.core.program_price(program) for name, program in programs.items()
    }
    result["stacked_typed_fit_nrmse"] = {
        name: runner._stacked_fit_nrmse(
            program, parent["prefilter_indices"], parent["prefilter_gram"],
            parent["prefilter_cross"], parent["native_typed_write_energy"],
        )
        for name, program in programs.items() if not name.startswith("affine_")
    }
    result["score_projection_replay_max_abs"] = {
        arm: float((
            runner.core.project_capped_simplex(score, 512) - score
        ).abs().max())
        for arm, score in artifact["scores"].items()
    }
    result["direct_polarization_replay"] = {
        name: runner.deployed_polarization_replay(program)
        for name, program in programs.items()
    }
    overlaps = {}
    for budget in runner.BUDGETS:
        real = set(artifact["supports"][f"teacher_k{budget}"].tolist())
        comparisons = {
            "teacher_row_reversal": artifact["supports"][
                f"teacher_row_reversal_k{budget}"
            ],
            "teacher_document_derangement": artifact["supports"][
                f"teacher_document_derangement_k{budget}"
            ],
            "random": artifact["supports"][f"random_k{budget}"],
            "family_A": family_a[budget],
        }
        for name, support in comparisons.items():
            other = set(support.tolist())
            overlaps[f"teacher_vs_{name}_k{budget}"] = {
                "intersection": len(real & other),
                "jaccard": len(real & other) / len(real | other),
            }
    result["support_overlaps"] = overlaps
    kwargs = {
        "expected_authority_sha256": "a" * 64,
        "expected_programs_file_sha256": "c" * 64,
        "program_artifact": artifact,
        "parent_payload": parent,
        "family_a_supports": family_a,
    }
    runner.semantic_validate_result(result, **kwargs)
    first = next(iter(result["stacked_typed_fit_nrmse"]))
    result["stacked_typed_fit_nrmse"][first] = 999.0
    with pytest.raises(RuntimeError, match="does not reconstruct"):
        runner.semantic_validate_result(result, **kwargs)


def test_receipt_semantic_replay_binds_all_terminal_artifacts():
    result = _valid_result()
    receipt = {
        "schema": "block3_consequence_family_f_v1_receipt",
        "status": "fit_complete_receipt_last_no_evaluation_opened",
        "authority_sha256": "a" * 64,
        "authority_file_sha256": "b" * 64,
        "programs_file_sha256": "c" * 64,
        "results_file_sha256": "d" * 64,
        "source_closure_sha256": "e" * 64,
        "prior_artifact_binding_sha256": "f" * 64,
        "row_binding_sha256": "1" * 64,
        "checkpoint_weights_sha256": "2" * 64,
        "call_ledger": result["call_ledger"],
        "validation_rows_loaded": 0, "final_rows_loaded": 0,
        "authorized_for_validation": False, "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False, "elapsed_seconds": 2.0,
    }
    kwargs = {
        "expected_authority_sha256": "a" * 64,
        "authority_file_sha256": "b" * 64,
        "programs_file_sha256": "c" * 64,
        "results_file_sha256": "d" * 64,
        "source_sha256": "e" * 64,
        "prior_sha256": "f" * 64,
        "rows_sha256": "1" * 64,
        "checkpoint_weights_sha256": "2" * 64,
        "expected_call_ledger": result["call_ledger"],
    }
    runner.semantic_validate_receipt(receipt, **kwargs)
    receipt["results_file_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="joins"):
        runner.semantic_validate_receipt(receipt, **kwargs)


def test_resource_ceiling_aborts_during_execution(monkeypatch):
    monkeypatch.setattr(runner.time, "time", lambda: runner.MAX_WALL_SECONDS + 1.0)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="resource ceiling"):
        runner.require_resource_ceiling(0.0)


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
    monkeypatch.setattr(
        runner, "semantic_validate_program_artifact", lambda value, **kwargs: None,
    )
    monkeypatch.setattr(runner, "semantic_validate_result", lambda value, **kwargs: None)
    monkeypatch.setattr(runner, "semantic_validate_receipt", lambda value, **kwargs: None)
    parent = {"prefilter_indices": torch.arange(runner.PREFILTER)}
    a_programs = {
        "programs": {
            f"activation_selected_k{budget}": {"indices": torch.arange(budget)}
            for budget in runner.BUDGETS
        }
    }
    monkeypatch.setattr(
        runner, "load_parent_tensors_after_authority",
        lambda frozen_authority, started: (parent, a_programs),
    )

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
