import json

import pytest
import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_inherited as inherited
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_observational_authority as authority
import early_mlp_suffix_transport_v1_observational_execution as execution
import early_mlp_suffix_transport_v1_runtime as runtime


def _frozen_rows(tmp_path, monkeypatch):
    paths = lifecycle.ArtifactPaths(root=tmp_path)
    paths.cache.mkdir(parents=True)
    fit_rows = (torch.arange(
        authority.FIT_ROW_COUNT * authority.FIT_ROW_WIDTH, dtype=torch.long,
    ).view(authority.FIT_ROW_COUNT, authority.FIT_ROW_WIDTH) % 101).contiguous()
    fit_cache = paths.cache / "fit.pt"
    torch.save({"rows": fit_rows}, fit_cache)
    manifest = {"kind": "synthetic frozen rows manifest"}
    paths.rows_manifest.write_text(json.dumps(manifest))
    receipt = {
        "entries": {
            authority.FIT_ROLE: {
                "cache_path": str(fit_cache),
                "cache_file_sha256": lifecycle.file_sha256(fit_cache),
                "tensor_full_raw_sha256": lifecycle.tensor_sha256(fit_rows),
                "shape_full": list(fit_rows.shape),
            },
        },
    }
    paths.rows_receipt.write_text(json.dumps(receipt))
    validations = []
    monkeypatch.setattr(
        lifecycle, "_validate_rows_receipt",
        lambda supplied, supplied_paths: validations.append((supplied, supplied_paths)),
    )
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 0)
    return paths, fit_rows, fit_cache, validations


def _denominator_pass() -> fit.DenominatorPass:
    count = capabilities.FIT_ROW_COUNT * (
        runtime.SCORE_STOP - runtime.SCORE_START
    )
    records = tuple({
        "count": count,
        "coordinate_sum": torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
        "coordinate_square_sum": torch.ones(runtime.CODE_DIM, dtype=torch.float64),
        "mean": torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
        "centered_sum_of_squares": torch.tensor(3.0 + site, dtype=torch.float64),
        "raw_sum_square_replay": torch.tensor(3.0 + site, dtype=torch.float64),
        "denominator": torch.tensor(2.0 + site, dtype=torch.float64),
        "ordered_support_sha256": "e" * 64,
    } for site in (0, 1))
    return fit.DenominatorPass(
        site_records=records, transaction_history_sha256="f" * 64,
        completed_steps=capabilities.FIT_BATCHES_PER_EPOCH,
    )


def _protected_denominator(tmp_path, monkeypatch):
    paths = lifecycle.ArtifactPaths(root=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    denominator = _denominator_pass()
    torch.save({
        authority.DENOMINATOR_LEDGER_KEY: authority.denominator_pass_payload(denominator),
        "unrelated_fit_state": torch.tensor([1.0]),
    }, paths.fit_ledger)
    paths.fit_manifest.write_text(json.dumps({"kind": "synthetic fit manifest"}))
    child = authority.denominator_pass_authority_payload(denominator, paths=paths)
    paths.fit_receipt.write_text(json.dumps({
        "kind": "synthetic fit receipt",
        authority.DENOMINATOR_AUTHORITY_KEY: child,
    }))
    protected = lifecycle.protected_snapshot((
        paths.fit_ledger, paths.fit_manifest, paths.fit_receipt,
    ))
    monkeypatch.setattr(
        lifecycle, "load_programs_unlock",
        lambda supplied_paths: {"protected_before": protected},
    )
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 0)
    return paths, denominator, protected


def test_reduced_loader_hash_validates_and_exposes_no_fit_rows(tmp_path, monkeypatch) -> None:
    paths, fit_rows, _cache, validations = _frozen_rows(tmp_path, monkeypatch)
    loaded = authority.load_fit_token_count_authority(paths=paths)
    targets = fit_rows[:, 65:257]
    expected = torch.bincount(
        targets.flatten(), minlength=execution.TOKEN_VOCAB,
    ).long()
    assert validations and validations[0][1] is paths
    assert loaded.receipt.fit_token_counts_sha256 == runtime.tensor_identity_sha256(expected)
    assert loaded.receipt.fit_target_count == targets.numel()
    assert all(not torch.is_tensor(getattr(loaded.receipt, name)) for name in (
        field for field in loaded.receipt.__dataclass_fields__
    ))
    assert not hasattr(loaded, "rows")
    assert lifecycle._FINAL_ROLE_LOADS == 0


def test_reduced_loader_rejects_cache_hash_drift_and_post_final_load(
    tmp_path, monkeypatch,
) -> None:
    paths, _rows, cache, _validations = _frozen_rows(tmp_path, monkeypatch)
    with cache.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(RuntimeError, match="cache binding"):
        authority.load_fit_token_count_authority(paths=paths)
    paths, _rows, _cache, _validations = _frozen_rows(
        tmp_path / "second", monkeypatch,
    )
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 1)
    with pytest.raises(RuntimeError, match="before the final role"):
        authority.load_fit_token_count_authority(paths=paths)


def test_reduced_authority_is_one_use_and_binds_exact_final_tensor(
    tmp_path, monkeypatch,
) -> None:
    paths, _fit_rows, _cache, _validations = _frozen_rows(tmp_path, monkeypatch)
    loaded = authority.load_fit_token_count_authority(paths=paths)
    final_rows = (torch.arange(192 * 513).view(192, 513) % 101).long().contiguous()
    context = capabilities.FinalRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(final_rows),
        identity_teacher_mapping_sha256="4" * 64,
    )
    plan = loaded.make_final_plan(final_rows=final_rows, final_context=context)
    assert plan.final_role_tensor_sha256 == context.final_role_tensor_sha256
    assert loaded.spent
    with pytest.raises(RuntimeError, match="already closed"):
        loaded.make_final_plan(final_rows=final_rows, final_context=context)


def test_executor_factory_spends_lifecycle_final_tensor_once(tmp_path, monkeypatch) -> None:
    paths, _fit_rows, _cache, _validations = _frozen_rows(tmp_path, monkeypatch)
    loaded = authority.load_fit_token_count_authority(paths=paths)
    final_rows = (torch.arange(192 * 513).view(192, 513) % 101).long().contiguous()
    context = capabilities.FinalRunContext(
        source_commit="5" * 40, inherited_snapshot_sha256="6" * 64,
        rows_receipt_sha256="7" * 64,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(final_rows),
        identity_teacher_mapping_sha256="8" * 64,
    )
    # Construction is tested independently by observational_execution.  Here the
    # sentinel isolates the authority/final-row handoff and its one-shot semantics.
    captured = []
    sentinel = object()

    def fake_executor(**kwargs):
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(execution, "FinalObservationalBatchExecutor", fake_executor)
    factory = authority.FinalObservationalExecutorFactory(
        final_context=context,
        inherited_initialization=object.__new__(authority.inherited.LoadedInitialization),
        denominator_pass=object.__new__(authority.fit.DenominatorPass),
        frequency_authority=loaded,
    )
    adapter = object.__new__(observed.ObservedBilin18Adapter)
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 1)
    result = factory.build(
        adapter=adapter, final_rows=final_rows,
        validated_program_bank={"payload_sha256": "9" * 64},
    )
    assert result is sentinel and len(captured) == 1
    assert captured[0]["final_rows"] is final_rows
    assert isinstance(captured[0]["frequency_plan"], execution.FinalFrequencyPlan)
    with pytest.raises(RuntimeError, match="already closed"):
        factory.build(
            adapter=adapter, final_rows=final_rows,
            validated_program_bank={"payload_sha256": "9" * 64},
        )


def test_executor_factory_poison_closes_on_substituted_final_rows(
    tmp_path, monkeypatch,
) -> None:
    paths, _fit_rows, _cache, _validations = _frozen_rows(tmp_path, monkeypatch)
    loaded = authority.load_fit_token_count_authority(paths=paths)
    final_rows = torch.zeros(192, 513, dtype=torch.long).contiguous()
    context = capabilities.FinalRunContext(
        source_commit="a" * 40, inherited_snapshot_sha256="b" * 64,
        rows_receipt_sha256="c" * 64,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(final_rows),
        identity_teacher_mapping_sha256="d" * 64,
    )
    factory = authority.FinalObservationalExecutorFactory(
        final_context=context,
        inherited_initialization=object.__new__(authority.inherited.LoadedInitialization),
        denominator_pass=object.__new__(authority.fit.DenominatorPass),
        frequency_authority=loaded,
    )
    changed = final_rows.clone()
    changed[0, 65] = 1
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 1)
    with pytest.raises(ValueError, match="final-role authority"):
        factory.build(
            adapter=object.__new__(observed.ObservedBilin18Adapter),
            final_rows=changed, validated_program_bank={},
        )
    with pytest.raises(RuntimeError, match="already closed"):
        factory.build(
            adapter=object.__new__(observed.ObservedBilin18Adapter),
            final_rows=final_rows, validated_program_bank={},
        )


def test_protected_denominator_round_trip_and_no_final_load(tmp_path, monkeypatch) -> None:
    paths, expected, _protected = _protected_denominator(tmp_path, monkeypatch)
    observed_pass = authority.load_protected_denominator_pass(paths=paths)
    assert observed_pass.sha256 == expected.sha256
    assert all(
        torch.equal(left, right) for left, right in zip(
            observed_pass.denominators, expected.denominators, strict=True,
        )
    )
    assert lifecycle._FINAL_ROLE_LOADS == 0


def test_protected_denominator_rejects_semantic_and_snapshot_drift(
    tmp_path, monkeypatch,
) -> None:
    paths, denominator, _protected = _protected_denominator(tmp_path, monkeypatch)
    ledger = torch.load(paths.fit_ledger, map_location="cpu", weights_only=True)
    ledger[authority.DENOMINATOR_LEDGER_KEY]["denominator_pass_sha256"] = "0" * 64
    torch.save(ledger, paths.fit_ledger)
    # Rebind the mocked protected snapshot to isolate semantic replay from byte drift.
    current = lifecycle.protected_snapshot((
        paths.fit_ledger, paths.fit_manifest, paths.fit_receipt,
    ))
    monkeypatch.setattr(
        lifecycle, "load_programs_unlock",
        lambda supplied_paths: {"protected_before": current},
    )
    with pytest.raises(RuntimeError, match="authority changed|identity changed"):
        authority.load_protected_denominator_pass(paths=paths)

    paths, denominator, protected = _protected_denominator(tmp_path / "drift", monkeypatch)
    paths.fit_manifest.write_text("drift")
    monkeypatch.setattr(
        lifecycle, "load_programs_unlock",
        lambda supplied_paths: {"protected_before": protected},
    )
    with pytest.raises(RuntimeError, match="omitted or changed|inputs drifted"):
        authority.load_protected_denominator_pass(paths=paths)


def test_final_context_reconstruction_binds_terminal_and_inherited_authorities() -> None:
    initialization = object.__new__(inherited.LoadedInitialization)
    inherited_authority = inherited.ValidatedInherited(
        bindings={}, ship_realization_sha256="1" * 64,
        compiler_source_commit="2" * 40, basis_source_commit="3" * 40,
        snapshot_sha256="4" * 64, full_product_sha256={0: "5" * 64, 1: "6" * 64},
    )
    object.__setattr__(
        initialization, "_LoadedInitialization__authority", inherited_authority,
    )
    rows_binding = {"path": "/tmp/rows.json", "sha256": "7" * 64, "bytes": 99}
    bindings = {"rows_receipt": rows_binding, "source_commit": "8" * 40}
    attempt = {
        "rows_receipt": dict(rows_binding), "source_commit": "8" * 40,
        "final_cache": {
            "shape_full": [192, 513], "tensor_full_raw_sha256": "9" * 64,
        },
    }
    context = authority.reconstruct_final_run_context(
        bindings=bindings, attempt=attempt,
        inherited_initialization=initialization,
    )
    assert context.source_commit == bindings["source_commit"]
    assert context.rows_receipt_sha256 == rows_binding["sha256"]
    assert context.final_role_tensor_sha256 == "9" * 64
    assert context.inherited_snapshot_sha256 == inherited_authority.snapshot_sha256
    assert context.identity_teacher_mapping_sha256 == (
        authority.identity_teacher_mapping_sha256(rows_binding["sha256"])
    )
    changed = dict(attempt)
    changed["rows_receipt"] = {**rows_binding, "bytes": 100}
    with pytest.raises(RuntimeError, match="authorities disagree"):
        authority.reconstruct_final_run_context(
            bindings=bindings, attempt=changed,
            inherited_initialization=initialization,
        )
