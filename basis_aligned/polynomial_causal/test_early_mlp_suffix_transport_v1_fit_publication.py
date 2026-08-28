import json

import pytest
import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_fit_publication as publication
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_observational_authority as final_authority
import early_mlp_suffix_transport_v1_runtime as runtime


def _denominator_pass() -> fit.DenominatorPass:
    count = capabilities.FIT_ROW_COUNT * (
        runtime.SCORE_STOP - runtime.SCORE_START
    )
    records = tuple({
        "count": count,
        "coordinate_sum": torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
        "coordinate_square_sum": torch.ones(runtime.CODE_DIM, dtype=torch.float64),
        "mean": torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
        "centered_sum_of_squares": torch.tensor(2.0 + site, dtype=torch.float64),
        "raw_sum_square_replay": torch.tensor(2.0 + site, dtype=torch.float64),
        "denominator": torch.tensor(1.0 + site, dtype=torch.float64),
        "ordered_support_sha256": "1" * 64,
    } for site in (0, 1))
    return fit.DenominatorPass(
        site_records=records, transaction_history_sha256="2" * 64,
        completed_steps=capabilities.FIT_BATCHES_PER_EPOCH,
    )


def _setup(tmp_path, monkeypatch):
    paths = lifecycle.ArtifactPaths(root=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths.rows_receipt.write_text(json.dumps({"kind": "frozen rows receipt"}))
    paths.rows_manifest.write_text(json.dumps({"kind": "frozen rows manifest"}))
    monkeypatch.setattr(lifecycle, "require_run_claim", lambda *args, **kwargs: None)
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda *args, **kwargs: None)
    source = {"source_commit": "3" * 40, "source_hashes": {"a.py": "4" * 64}}
    before = lifecycle.protected_snapshot((paths.rows_receipt, paths.rows_manifest))
    return paths, source, before


def _publish(tmp_path, monkeypatch):
    paths, source, before = _setup(tmp_path, monkeypatch)
    denominator = _denominator_pass()
    result = publication.publish_fit_artifacts(
        fit_records={
            "candidate_receipts": ["5" * 64, "6" * 64],
            "loss_curve": torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        },
        manifest_records={"candidate_count": 2, "routes": ["L", "R"]},
        denominator_pass=denominator,
        fit_execution_sha256="7" * 64, fit_role_tensor_sha256="8" * 64,
        source_closure=source, protected_before=before, lock_nonce="nonce",
        paths=paths, lock_path=tmp_path / "lock",
    )
    return paths, denominator, result


def test_transactional_fit_publication_injects_and_replays_denominator(
    tmp_path, monkeypatch,
) -> None:
    order = []
    real_torch = lifecycle.atomic_create_torch
    real_json = lifecycle.atomic_create_json

    def write_torch(value, path):
        order.append(path.name)
        return real_torch(value, path)

    def write_json(value, path):
        order.append(path.name)
        return real_json(value, path)

    monkeypatch.setattr(lifecycle, "atomic_create_torch", write_torch)
    monkeypatch.setattr(lifecycle, "atomic_create_json", write_json)
    paths, denominator, (ledger, manifest, receipt) = _publish(tmp_path, monkeypatch)
    assert order == [paths.fit_ledger.name, paths.fit_manifest.name, paths.fit_receipt.name]
    assert ledger["denominator_pass"].sha256 == denominator.sha256
    assert manifest["denominator_pass_sha256"] == denominator.sha256
    child = receipt[final_authority.DENOMINATOR_AUTHORITY_KEY]
    assert child["denominator_pass_sha256"] == denominator.sha256
    assert receipt["authorized_for_selection"] is True
    assert publication.validate_fit_publication(paths=paths)[2] == receipt


def test_published_denominator_is_consumable_from_program_protected_snapshot(
    tmp_path, monkeypatch,
) -> None:
    paths, denominator, _result = _publish(tmp_path, monkeypatch)
    protected = lifecycle.protected_snapshot((
        paths.fit_ledger, paths.fit_manifest, paths.fit_receipt,
    ))
    monkeypatch.setattr(
        lifecycle, "load_programs_unlock",
        lambda supplied_paths: {"protected_before": protected},
    )
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 0)
    restored = final_authority.load_protected_denominator_pass(paths=paths)
    assert restored.sha256 == denominator.sha256


def test_fit_publication_rejects_graph_tensor_and_create_only_reentry(
    tmp_path, monkeypatch,
) -> None:
    paths, source, before = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="detached CPU"):
        publication.publish_fit_artifacts(
            fit_records={"bad": torch.ones(1, requires_grad=True)},
            manifest_records={}, denominator_pass=_denominator_pass(),
            fit_execution_sha256="9" * 64, fit_role_tensor_sha256="a" * 64,
            source_closure=source, protected_before=before, lock_nonce="nonce",
            paths=paths, lock_path=tmp_path / "lock",
        )
    assert not paths.fit_ledger.exists()
    _publish(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="ordering failed"):
        publication.publish_fit_artifacts(
            fit_records={"again": 1}, manifest_records={},
            denominator_pass=_denominator_pass(),
            fit_execution_sha256="9" * 64, fit_role_tensor_sha256="a" * 64,
            source_closure=source, protected_before=before, lock_nonce="nonce",
            paths=paths, lock_path=tmp_path / "lock",
        )


def test_manifest_failure_never_publishes_selection_receipt(tmp_path, monkeypatch) -> None:
    paths, source, before = _setup(tmp_path, monkeypatch)
    real_json = lifecycle.atomic_create_json

    def fail_manifest(value, path):
        if path == paths.fit_manifest:
            raise RuntimeError("injected manifest failure")
        return real_json(value, path)

    monkeypatch.setattr(lifecycle, "atomic_create_json", fail_manifest)
    with pytest.raises(RuntimeError, match="injected manifest failure"):
        publication.publish_fit_artifacts(
            fit_records={"safe": torch.ones(1)}, manifest_records={},
            denominator_pass=_denominator_pass(),
            fit_execution_sha256="b" * 64, fit_role_tensor_sha256="c" * 64,
            source_closure=source, protected_before=before, lock_nonce="nonce",
            paths=paths, lock_path=tmp_path / "lock",
        )
    assert paths.fit_ledger.is_file()
    assert not paths.fit_manifest.exists() and not paths.fit_receipt.exists()


def test_receipt_or_ledger_drift_fails_semantic_reload(tmp_path, monkeypatch) -> None:
    paths, _denominator, _result = _publish(tmp_path, monkeypatch)
    receipt = json.loads(paths.fit_receipt.read_text())
    receipt[final_authority.DENOMINATOR_AUTHORITY_KEY][
        "denominator_pass_sha256"
    ] = "d" * 64
    paths.fit_receipt.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="denominator child"):
        publication.validate_fit_publication(paths=paths)
