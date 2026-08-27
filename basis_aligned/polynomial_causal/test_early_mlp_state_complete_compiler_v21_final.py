from __future__ import annotations

import ast
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch

import early_mlp_state_complete_compiler_v21_final as final


def test_lattice_arm_contract_and_original_call_allowlist() -> None:
    states, allowed = final._states_for_arm("QOE")
    assert states == {0: "Q", 1: "O", 2: "E"}
    assert allowed == {1, 2}
    assert final._states_for_arm("NNN") == ({}, set())
    with pytest.raises(ValueError):
        final._states_for_arm("QQQ")


def test_document_bootstrap_keeps_rows_from_one_document_paired() -> None:
    documents = [f"d{i // 2}" for i in range(192)]
    weights, unique = final.document_bootstrap_weights(documents, draws=25, seed=7)
    assert weights.shape == (25, 192)
    assert len(unique) == 96
    assert torch.equal(weights[:, 0], weights[:, 1])
    assert torch.equal(weights[:, 190], weights[:, 191])
    assert torch.equal(weights.sum(dim=1), torch.full((25,), 192.0))


def test_series_preserves_exact_paired_zero_difference() -> None:
    documents = [f"d{i}" for i in range(192)]
    weights, _ = final.document_bootstrap_weights(documents, draws=40, seed=11)
    numerator = torch.arange(1, 193, dtype=torch.float64)
    denominator = torch.full((192,), 192, dtype=torch.long)
    left = final._series(numerator, denominator, weights)
    right = final._series(numerator.clone(), denominator.clone(), weights)
    assert torch.equal(left - right, torch.zeros_like(left))


def test_holm_is_step_down_and_one_sided() -> None:
    positive = torch.cat([torch.tensor([1.0]), torch.ones(2000)])
    mixed = torch.cat([torch.tensor([0.1]), torch.linspace(-1, 1, 2000)])
    result = final._holm_positive({"positive": positive, "mixed": mixed})
    assert result["rejected_positive"]["positive"] is True
    assert result["rejected_positive"]["mixed"] is False


@pytest.mark.parametrize("grammar", ["affine", "native", "constant"])
def test_signed_gauge_preserves_physical_state_complete_correction(grammar) -> None:
    generator = torch.Generator().manual_seed(3)
    basis = torch.randn(1152, 64, generator=generator)
    z = torch.randn(2, 1152, generator=generator)
    mo = torch.randn(2, 1152, generator=generator)
    if grammar == "affine":
        state = {
            "grammar": "affine", "interface": "state_complete_p",
            "mean": torch.zeros(1152), "scale": torch.ones(1152),
            "left": torch.randn(1152, 2, generator=generator),
            "right": torch.randn(2, 64, generator=generator),
            "bias": torch.randn(64, generator=generator),
        }
    elif grammar == "native":
        state = {
            "grammar": "native", "interface": "state_complete_p",
            "left": torch.randn(2, 1152, generator=generator),
            "right": torch.randn(2, 1152, generator=generator),
            "projected_decoder": torch.randn(2, 64, generator=generator),
            "beta": torch.randn(64, generator=generator),
        }
    else:
        state = {
            "grammar": "constant", "interface": "state_complete_p",
            "bias": torch.randn(64, generator=generator),
        }
    signs = torch.where(torch.arange(64) % 2 == 0, 1.0, -1.0)
    moved = final._signed_gauge_state(state, signs)
    before = final.old_site0.runtime.runtime_coefficients(z, mo, basis, state) @ basis.T
    moved_basis = basis * signs
    after = final.old_site0.runtime.runtime_coefficients(
        z, mo, moved_basis, moved,
    ) @ moved_basis.T
    assert torch.allclose(before, after, atol=2e-4, rtol=2e-5)


def _valid_raw() -> dict:
    count = torch.full((192,), 192, dtype=torch.long)
    frequency_count = torch.zeros(192, 9, dtype=torch.long)
    frequency_count[:, 0] = 192
    return {
        "row_ce_sum": torch.ones(192, dtype=torch.float64),
        "row_ce_count": count,
        "row_copy_ce_sum": torch.ones(192, dtype=torch.float64),
        "row_copy_count": torch.ones(192, dtype=torch.long),
        "row_frequency_ce_sum": torch.ones(192, 9, dtype=torch.float64),
        "row_frequency_count": frequency_count,
        "row_teacher_kl_sum": {},
        "row_teacher_kl_count": {},
    }


def test_arm_statistics_require_exact_frequency_partition() -> None:
    raw = _valid_raw()
    final.validate_arm_statistics(raw)
    raw["row_frequency_count"][0, 0] = 191
    with pytest.raises(RuntimeError, match="supports do not partition"):
        final.validate_arm_statistics(raw)


def test_attempt_is_written_before_final_loader_in_main_source() -> None:
    source = Path(final.__file__).read_text()
    tree = ast.parse(source)
    main = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    )
    rendered = ast.unparse(main)
    assert rendered.index("write_attempt_before_final_load") < rendered.index(
        "load_final_for_scoring"
    )
    assert rendered.rindex("write_json_atomic(outcome, OUTCOME_AUTHORITY)") > rendered.index(
        "write_json_atomic(manifest, MANIFEST)"
    )


def test_attempt_record_binds_final_without_deserializing_it(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / name for name in ("attempt.json", "result.pt", "manifest.json", "authority.json")]
    monkeypatch.setattr(final, "ATTEMPT", paths[0])
    monkeypatch.setattr(final, "RESULT", paths[1])
    monkeypatch.setattr(final, "MANIFEST", paths[2])
    monkeypatch.setattr(final, "OUTCOME_AUTHORITY", paths[3])
    monkeypatch.setattr(final, "FINAL_OUTPUTS", tuple(paths))
    final_cache = tmp_path / "final.pt"
    final_cache.write_bytes(b"not deserialized")
    bindings = {}
    for name in ("unlock", "program", "rows", "rows_manifest"):
        path = tmp_path / name
        path.write_text(name)
        bindings[name] = path
    monkeypatch.setattr(final.authority, "PROGRAMS_RECEIPT", bindings["unlock"])
    monkeypatch.setattr(final.authority, "PROGRAMS_ARTIFACT", bindings["program"])
    monkeypatch.setattr(final.authority, "RECEIPT", bindings["rows"])
    monkeypatch.setattr(final.authority, "MANIFEST", bindings["rows_manifest"])
    monkeypatch.setattr(final, "_final_entry", lambda: (
        {
            "cache_file_sha256": final.authority.file_sha256(final_cache),
            "tensor_full_raw_sha256": "a" * 64,
            "tensor_prefix257_raw_sha256": "b" * 64,
            "provenance_records_sha256": "c" * 64,
        },
        {"records": [], "path": final_cache},
    ))
    attempt = final.write_attempt_before_final_load(
        {"source_commit": "d" * 40, "source_hashes": {"x": "e" * 64}},
        protected_before={"p": None},
    )
    assert json.loads(paths[0].read_text()) == attempt
    assert attempt["requested_role"] == "compiler_final_v21"
    assert attempt["authority"] == "none"


def test_final_rulings_are_hash_pinned() -> None:
    assert final.authority.file_sha256(final.authority.FINAL_RULINGS) == (
        final.authority.FINAL_RULINGS_SHA256
    )


def _analysis_arm(name: str, ce: float = 1.0) -> dict:
    count = torch.full((192,), 192, dtype=torch.long)
    frequency_count = torch.zeros(192, 9, dtype=torch.long)
    frequency_count[:, 0] = 191
    frequency_count[0, 1] = 1
    frequency_count[0, 0] = 190
    teachers = set() if name == "OON" else ({"OON"} if name == "QON" else {"OON", "QON"})
    return {
        "name": name, "scorer": final.SCORER,
        "row_ce_sum": torch.full((192,), ce * 192, dtype=torch.float64),
        "row_ce_count": count,
        "row_copy_ce_sum": torch.full((192,), ce, dtype=torch.float64),
        "row_copy_count": torch.tensor([1] + [0] * 191, dtype=torch.long),
        "row_frequency_ce_sum": frequency_count.double() * ce,
        "row_frequency_count": frequency_count,
        "row_teacher_kl_sum": {
            teacher: torch.full((192,), 192.0, dtype=torch.float64)
            for teacher in teachers
        },
        "row_teacher_kl_count": {teacher: count.clone() for teacher in teachers},
        "original_mlp_call_counters": {0: 0, 1: 0, 2: 0},
    }


def test_sparse_collateral_is_authoritative_negative_not_abort() -> None:
    arms = {name: _analysis_arm(name) for name in final._expected_arm_names()}
    bundle = {
        "prices": {
            "true": {"total_reals": 1}, "shuffle": {"total_reals": 2},
        }
    }
    report = final.analyze(arms, [f"doc-{i}" for i in range(192)], bundle)
    assert report["registered_gates"]["copy_collateral"] is False
    assert report["registered_gates"]["frequency_collateral"] is False
    assert report["package_admitted"] is False
    assert report["copy_worsening"]["QQN_vs_NNN"]["ci_status"] == (
        "unevaluable_zero_support_resamples"
    )
    assert report["frequency_collateral"]["QQN_vs_NNN"][1]["worsening"][
        "bootstrap_zero_support_draws"
    ] > 0


def _patch_final_main(monkeypatch, tmp_path, *, fail_validation_call: int | None = None):
    paths = [tmp_path / name for name in (
        "attempt.json", "result.pt", "manifest.json", "authority.json",
    )]
    for name, path in zip(
        ("ATTEMPT", "RESULT", "MANIFEST", "OUTCOME_AUTHORITY"), paths, strict=True,
    ):
        monkeypatch.setattr(final, name, path)
    monkeypatch.setattr(final, "FINAL_OUTPUTS", tuple(paths))
    unlock_path, bundle_path = tmp_path / "unlock.json", tmp_path / "bundle.pt"
    unlock_path.write_text("unlock")
    torch.save({}, bundle_path)
    monkeypatch.setattr(final.authority, "PROGRAMS_RECEIPT", unlock_path)
    monkeypatch.setattr(final.authority, "PROGRAMS_ARTIFACT", bundle_path)
    unlock = {"source_commit": "a" * 40, "source_hashes": {"x": "b" * 64}}
    monkeypatch.setattr(final.lifecycle, "exclusive_run_claim", lambda: nullcontext())
    monkeypatch.setattr(final.authority, "validate_final_unlock", lambda _: unlock)
    monkeypatch.setattr(final.authority, "protected_snapshot", lambda: {"p": "q"})
    attempt = {
        "final_cache": {"sha256": "c" * 64}, "source_commit": "a" * 40,
        "source_hashes": {"x": "b" * 64},
    }
    def write_attempt(*_args, **_kwargs):
        final.authority.write_json_atomic(attempt, final.ATTEMPT)
        return attempt
    monkeypatch.setattr(final, "write_attempt_before_final_load", write_attempt)
    monkeypatch.setattr(final.authority, "load_final_for_scoring", lambda _: ({}, torch.zeros(1)))
    result = {
        "integrity": True,
        "analysis": {"package_admitted": False, "claim_scope": "negative"},
        "execution_closure": {"outer_model_returned": True},
        "diagnostics": {"final_role_loads": 1, "final_evaluation_callbacks": 1},
    }
    monkeypatch.setattr(
        final, "_run_final", lambda *_args, **_kwargs: (result, {"p": "q"}),
    )
    calls = {"validation": 0}
    def validate(*_args, **_kwargs):
        calls["validation"] += 1
        if calls["validation"] == fail_validation_call:
            raise RuntimeError("synthetic semantic corruption")
    monkeypatch.setattr(final, "validate_final_result", validate)
    return paths, calls


def test_final_main_success_writes_authority_last(monkeypatch, tmp_path) -> None:
    paths, calls = _patch_final_main(monkeypatch, tmp_path)
    final.main()
    assert calls["validation"] == 2
    assert all(path.is_file() for path in paths)
    outcome = json.loads(paths[3].read_text())
    assert outcome["status"] == "authoritative_negative_v21_final"
    assert outcome["manifest"] == final._binding(paths[2])


def test_final_main_semantic_result_failure_preserves_result_without_authority(
    monkeypatch, tmp_path,
) -> None:
    paths, _ = _patch_final_main(monkeypatch, tmp_path, fail_validation_call=1)
    with pytest.raises(RuntimeError, match="synthetic semantic corruption"):
        final.main()
    assert paths[0].is_file() and paths[1].is_file() and paths[2].is_file()
    assert not paths[3].exists()
    failure = json.loads(paths[2].read_text())
    assert failure["status"] == "failed_v21_final_without_outcome_authority"
    assert failure["preserved_outputs"]["result"] == final._binding(paths[1])


def test_attempt_postwrite_failure_gets_failure_manifest(monkeypatch, tmp_path) -> None:
    paths, _ = _patch_final_main(monkeypatch, tmp_path)
    def fail_after_attempt(*_args, **_kwargs):
        final.authority.write_json_atomic({"final_cache": {}}, final.ATTEMPT)
        raise RuntimeError("attempt reload failed")
    monkeypatch.setattr(final, "write_attempt_before_final_load", fail_after_attempt)
    with pytest.raises(RuntimeError, match="attempt reload failed"):
        final.main()
    assert paths[0].is_file() and paths[2].is_file() and not paths[3].exists()


def test_stage_fit_numerics_bind_selection_and_recompute_condition_ladder() -> None:
    condition = {
        "matrix": "normalized fit Gram plus lambda I",
        "rows": final.authority.FIT_CAPTURE_COUNT,
        "columns": final.compiler.D_MODEL,
        "minimum_gram_eigenvalue": 1.0,
        "maximum_gram_eigenvalue": 3.0,
        "condition_number_by_lambda": {
            str(float(ridge)): {
                "status": "evaluated", "value": (3.0 + ridge) / (1.0 + ridge),
            } for ridge in final.old_site0.affine_v1.LAMBDA_GRID
        },
    }
    replay = {
        "status": "evaluated_serialized_float32_parameters",
        "support_positions": 64, "reference": "float64 accumulation",
        "deployed": "float32 accumulation", "max_abs_coefficient_drift": 1e-6,
        "rms_coefficient_drift": 1e-7,
    }
    bundle = {
        "programs": {
            arm: {0: {"grammar": "constant"}} for arm in ("true", "shuffle", "mean")
        },
        "selection_receipts": {
            "true_site0": {"selected": "true-cell"},
            "shuffle_site0": {"selected": "shuffle-cell"},
        },
    }
    manifest = {
        "selected": {"true": "true-cell", "shuffle": "shuffle-cell"},
        "selected_fit_numerics": {},
    }
    for arm, name in (("true", "true-cell"), ("shuffle", "shuffle-cell"), ("mean", "mean_site0")):
        manifest["selected_fit_numerics"][arm] = {
            "selected": name, "grammar": "constant",
            "ridge_condition_numbers": condition,
            "float64_to_float32_replay": replay,
            "quantization_status": (
                "none; all floating parameter tensors float32; native indices int64"
            ),
        }
    final._validate_selected_fit_numerics(manifest, site=0, bundle=bundle)
    manifest["selected_fit_numerics"]["true"]["selected"] = "wrong"
    with pytest.raises(RuntimeError, match="binding changed"):
        final._validate_selected_fit_numerics(manifest, site=0, bundle=bundle)
