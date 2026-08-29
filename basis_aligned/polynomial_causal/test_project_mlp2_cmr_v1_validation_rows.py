from __future__ import annotations

import io
import inspect
from pathlib import Path

import pytest
import torch

import project_mlp2_cmr_v1_validation_rows as projection


def synthetic_role() -> dict[str, torch.Tensor]:
    clipped = torch.tensor([222] * 190 + [139, 64], dtype=torch.long)
    rows = torch.full(
        (projection.DOCUMENTS, projection.WIDTH), projection.EOT, dtype=torch.long,
    )
    for ordinal, count in enumerate(clipped.tolist()):
        rows[ordinal, :count] = torch.arange(count, dtype=torch.long).remainder(50_000)
    positions = torch.arange(projection.SEQUENCE).unsqueeze(0)
    eligible = (positions >= projection.SCORE_START) & (
        positions < (clipped - 1).clamp_min(0)[:, None]
    )
    return {
        "document_indices": torch.arange(projection.DOCUMENTS, dtype=torch.long),
        "rows": rows,
        "eligible_mask": eligible,
        "original_token_counts": clipped.clone(),
        "clipped_token_counts": clipped,
    }


def test_validation_semantics_exact_without_opening_real_role() -> None:
    role = synthetic_role()
    summary = projection.validate_role(role, require_identity=False)
    assert summary["documents"] == 192
    assert summary["eligible_positions"] == 29_904
    assert summary["support_documents"] == 191
    assert summary["all_false_ordinals"] == [191]
    assert not bool(role["eligible_mask"][:, :64].any())


def test_validation_role_rejects_wrong_census_padding_and_extra_keys() -> None:
    role = synthetic_role()
    role["eligible_mask"][0, 0] = True
    with pytest.raises(RuntimeError, match="eligibility"):
        projection.validate_role(role, require_identity=False)
    role = synthetic_role()
    role["rows"][191, 100] = 3
    with pytest.raises(RuntimeError, match="padding"):
        projection.validate_role(role, require_identity=False)
    role = synthetic_role()
    role["REPLICATION"] = torch.tensor(1)
    with pytest.raises(RuntimeError, match="keys"):
        projection.validate_role(role, require_identity=False)


def test_projector_is_model_free_and_publishes_only_validation() -> None:
    source = inspect.getsource(projection)
    assert "load_bilin18" not in source
    assert 'combined["VALIDATION"]' in source
    assert 'contains_roles": ["VALIDATION"]' in source
    assert 'authorized_for_validation_model_forward_input": True' in source
    assert 'authorized_for_replication": False' in source
    assert 'projection_loaded_model": False' in source
    assert "write_create_only_guarded(" in source
    assert "before_link=receipt_guard" in source
    assert set(projection.EXPECTED_TENSOR_HASHES) == {
        "document_indices", "rows", "eligible_mask", "original_token_counts",
        "clipped_token_counts",
    }


def test_source_closure_contains_contract_projector_and_tests() -> None:
    assert len(projection.SOURCE_CLOSURE) == len(set(projection.SOURCE_CLOSURE))
    names = {path.name for path in projection.SOURCE_CLOSURE}
    assert names == {
        "MLP2_CMR_V1_PREREGISTRATION.md",
        "MLP2_CMR_V1_VALIDATION_ADDENDUM.md",
        "project_mlp2_cmr_v1_validation_rows.py",
        "test_project_mlp2_cmr_v1_validation_rows.py",
        "project_mlp2_cmr_v1_fit_selector_rows.py",
        "test_project_mlp2_cmr_v1_fit_selector_rows.py",
        "MLP2_CMR_V1_MARGIN_FREQUENCY_ADDENDUM.md",
        "materialize_mlp2_cmr_v1_token_rows.py",
        "test_materialize_mlp2_cmr_v1_token_rows.py",
    }


def _transaction_fixture(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    monkeypatch.setattr(projection, "ROOT", tmp_path)
    paths = {
        name: tmp_path / name for name in (
            "output.pt", "manifest.json", "receipt.json", "failure.json", "lock",
        )
    }
    for attribute, key in (
        ("OUTPUT", "output.pt"), ("MANIFEST", "manifest.json"),
        ("RECEIPT", "receipt.json"), ("FAILURE", "failure.json"), ("LOCK", "lock"),
    ):
        monkeypatch.setattr(projection, attribute, paths[key])
    combined = {
        role: {"marker": torch.tensor([ordinal])}
        for ordinal, role in enumerate(("FIT_MEAN", "FIT_SELECTOR", "VALIDATION", "REPLICATION"))
    }
    stream = io.BytesIO()
    torch.save(combined, stream)
    captured = {"combined": stream.getvalue()}
    parents = {"combined": "parent"}
    monkeypatch.setattr(projection, "committed_source", lambda: ("commit", {}))
    monkeypatch.setattr(projection.base, "parent_snapshot", lambda: (parents, captured))
    monkeypatch.setattr(
        projection, "validate_role",
        lambda role: {"marker": int(role["marker"].item())},
    )
    return paths


def test_project_transaction_success(monkeypatch, tmp_path: Path) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    receipt = projection.project()
    assert receipt["status"] == "validation_role_only_projection_complete_receipt_last"
    assert paths["receipt.json"].exists() and not paths["failure.json"].exists()
    output = torch.load(paths["output.pt"], map_location="cpu", weights_only=True)
    assert set(output) == {"marker"} and int(output["marker"].item()) == 2


def test_project_parent_drift_and_lock_replacement_publish_no_terminal(
    monkeypatch, tmp_path: Path,
) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    calls = 0

    # Preserve the captured first input, then drift only the terminal recheck.
    original_parent = projection.base.parent_snapshot
    parents, captured = original_parent()
    monkeypatch.setattr(
        projection.base, "parent_snapshot",
        lambda: (parents, captured) if calls == 0 else ({"combined": "changed"}, captured),
    )
    calls = 0
    original_guarded = projection.write_create_only_guarded

    def drift_before_receipt(path, data, *, before_link):
        nonlocal calls
        if path == paths["receipt.json"]:
            calls = 1
        return original_guarded(path, data, before_link=before_link)

    monkeypatch.setattr(projection, "write_create_only_guarded", drift_before_receipt)
    with pytest.raises(RuntimeError, match="terminal snapshot"):
        projection.project()
    assert not paths["receipt.json"].exists() and not paths["failure.json"].exists()

    second = tmp_path / "lock_case"
    second.mkdir()
    paths = _transaction_fixture(monkeypatch, second)
    original_publish = projection.base.publish_torch_create_only

    def replace_lock(path, value):
        original_publish(path, value)
        paths["lock"].unlink()
        paths["lock"].write_text('{"nonce":"replacement"}')

    monkeypatch.setattr(projection.base, "publish_torch_create_only", replace_lock)
    with pytest.raises(RuntimeError, match="claim changed"):
        projection.project()
    assert not paths["receipt.json"].exists() and not paths["failure.json"].exists()

    third = tmp_path / "terminal_snapshot_lock_case"
    third.mkdir()
    paths = _transaction_fixture(monkeypatch, third)
    stable_parent = projection.base.parent_snapshot
    parents, captured = stable_parent()
    snapshots = 0

    def replace_lock_during_terminal_snapshot():
        nonlocal snapshots
        snapshots += 1
        if snapshots == 2:
            paths["lock"].unlink()
            paths["lock"].write_text('{"nonce":"late-replacement"}')
        return parents, captured

    monkeypatch.setattr(
        projection.base, "parent_snapshot", replace_lock_during_terminal_snapshot,
    )
    monkeypatch.setattr(
        projection.base, "publish_torch_create_only",
        original_publish,
    )
    with pytest.raises(RuntimeError, match="claim changed"):
        projection.project()
    assert not paths["receipt.json"].exists() and not paths["failure.json"].exists()


def test_corrupt_reload_publishes_failure_only(monkeypatch, tmp_path: Path) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    original_load = projection.torch.load
    calls = 0

    def corrupt_second_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return {"marker": torch.tensor([99])}
        return original_load(*args, **kwargs)

    monkeypatch.setattr(projection.torch, "load", corrupt_second_load)
    with pytest.raises(RuntimeError, match="semantic replay"):
        projection.project()
    assert paths["failure.json"].exists() and not paths["receipt.json"].exists()


def test_bidirectional_late_terminal_races_leave_only_one_terminal(
    monkeypatch, tmp_path: Path,
) -> None:
    paths = _transaction_fixture(monkeypatch, tmp_path)
    original_guarded = projection.write_create_only_guarded

    def failure_wins(path, data, *, before_link):
        if path == paths["receipt.json"]:
            projection.base.write_create_only(paths["failure.json"], b"{}")
        return original_guarded(path, data, before_link=before_link)

    monkeypatch.setattr(projection, "write_create_only_guarded", failure_wins)
    with pytest.raises(RuntimeError, match="terminal snapshot"):
        projection.project()
    assert paths["failure.json"].exists() and not paths["receipt.json"].exists()

    second = tmp_path / "receipt_case"
    second.mkdir()
    paths = _transaction_fixture(monkeypatch, second)
    original_load = projection.torch.load
    loads = 0

    def corrupt_second_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        if loads == 2:
            return {"marker": torch.tensor([99])}
        return original_load(*args, **kwargs)

    monkeypatch.setattr(projection.torch, "load", corrupt_second_load)
    original_guarded = projection.write_create_only_guarded

    def receipt_wins(path, data, *, before_link):
        if path == paths["failure.json"]:
            projection.base.write_create_only(paths["receipt.json"], b"{}")
        return original_guarded(path, data, before_link=before_link)

    monkeypatch.setattr(projection, "write_create_only_guarded", receipt_wins)
    with pytest.raises(RuntimeError, match="semantic replay"):
        projection.project()
    assert paths["receipt.json"].exists() and not paths["failure.json"].exists()
