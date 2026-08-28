from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

import early_mlp_suffix_transport_v1_lifecycle as lifecycle


def _touch(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_candidate_triples_match_frozen_schedule() -> None:
    zero = lifecycle.candidate_triple(0)
    two = lifecycle.candidate_triple(2)
    assert (zero.fit_n, zero.fit_skip) == (384, 43000)
    assert (zero.validation_n, zero.validation_skip) == (192, 47000)
    assert (zero.final_n, zero.final_skip) == (192, 51000)
    assert (two.fit_skip, two.validation_skip, two.final_skip) == (
        67000, 71000, 75000,
    )
    with pytest.raises(ValueError):
        lifecycle.candidate_triple(-1)


def test_canonical_namespace_is_complete_and_unique(tmp_path: Path) -> None:
    paths = lifecycle.ArtifactPaths(tmp_path)
    outputs = paths.output_files()
    assert len(outputs) == 13
    assert len(set(outputs)) == len(outputs)
    assert paths.cache.name == ".rowcache_early_mlp_suffix_transport_v1"
    assert paths.rows_receipt.name == "early_mlp_suffix_transport_v1_rows_receipt.json"
    assert paths.fit_ledger.name == "early_mlp_suffix_transport_v1_fit_ledger.pt"
    assert paths.final_authority.name == "early_mlp_suffix_transport_v1_final_authority.json"


def test_stage_ordering_fails_closed(tmp_path: Path) -> None:
    paths = lifecycle.ArtifactPaths(tmp_path)
    paths.assert_stage_preconditions("rows")
    _touch(paths.rows_manifest)
    with pytest.raises(RuntimeError, match="ordering"):
        paths.assert_stage_preconditions("rows")
    with pytest.raises(RuntimeError, match="missing"):
        paths.assert_stage_preconditions("fit")
    _touch(paths.rows_receipt)
    paths.assert_stage_preconditions("fit")
    _touch(paths.fit_ledger)
    _touch(paths.fit_manifest)
    _touch(paths.fit_receipt)
    paths.assert_stage_preconditions("programs")
    _touch(paths.programs)
    _touch(paths.programs_receipt)
    paths.assert_stage_preconditions("final_attempt")


def test_atomic_publication_is_create_only(tmp_path: Path) -> None:
    json_path = tmp_path / "record.json"
    lifecycle.atomic_create_json({"a": 1}, json_path)
    with pytest.raises(RuntimeError, match="overwrite"):
        lifecycle.atomic_create_json({"a": 2}, json_path)
    assert json.loads(json_path.read_text()) == {"a": 1}

    torch_path = tmp_path / "record.pt"
    lifecycle.atomic_create_torch({"x": torch.arange(3)}, torch_path)
    with pytest.raises(RuntimeError, match="overwrite"):
        lifecycle.atomic_create_torch({"x": torch.arange(4)}, torch_path)
    assert torch.equal(
        torch.load(torch_path, weights_only=True)["x"], torch.arange(3),
    )


def test_exclusive_lock_is_owned_and_create_only(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    with lifecycle.exclusive_run_claim(lock) as nonce:
        lifecycle.require_run_claim(nonce, lock)
        with pytest.raises(RuntimeError, match="already claimed"):
            with lifecycle.exclusive_run_claim(lock):
                pass
    assert not lock.exists()

    with pytest.raises(RuntimeError, match="not owned"):
        with lifecycle.exclusive_run_claim(lock) as nonce:
            lifecycle.require_run_claim(nonce, lock)
            lock.write_text("stolen")
    # A stolen claim is not removed by the former owner.
    assert lock.read_text() == "stolen"


def test_collision_report_checks_candidate_pairs_but_ignores_prior_internal_overlap() -> None:
    prior_a = lifecycle.RoleIdentity(
        frozenset({"old-doc"}), frozenset({"old-row"}), frozenset({"old-prefix"}),
    )
    prior_b = lifecycle.RoleIdentity(
        frozenset({"old-doc"}), frozenset({"other-row"}), frozenset({"other-prefix"}),
    )
    clean = lifecycle.RoleIdentity(
        frozenset({"new-doc"}), frozenset({"new-row"}), frozenset({"new-prefix"}),
    )
    report = lifecycle.collision_report(
        {"fit": clean}, {"prior_a": prior_a, "prior_b": prior_b},
    )
    assert report == {"collision_free": True, "collisions": []}

    collision = lifecycle.RoleIdentity(
        frozenset({"old-doc"}), frozenset({"new-row-2"}), frozenset({"new-prefix-2"}),
    )
    report = lifecycle.collision_report(
        {"fit": clean, "validation": collision}, {"prior_a": prior_a},
    )
    assert not report["collision_free"]
    assert report["collisions"][0]["documents"] == 1


def _row_entry(path: Path, rows: torch.Tensor) -> dict[str, object]:
    torch.save(rows, path)
    return {
        "cache_path": str(path),
        "cache_file_sha256": lifecycle.file_sha256(path),
        "shape_full": list(rows.shape),
        "tensor_full_raw_sha256": lifecycle.tensor_sha256(rows),
    }


def _write_row_receipt(paths: lifecycle.ArtifactPaths):
    tensors = {
        role: torch.full((2, 33), index, dtype=torch.long)
        for index, role in enumerate(lifecycle.ROLE_NAMES)
    }
    entries = {
        role: _row_entry(paths.root / f"rows-{index}.pt", tensor)
        for index, (role, tensor) in enumerate(tensors.items())
    }
    receipt = {
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "role_licenses": lifecycle.ROLE_LICENSES,
        "entries": entries,
    }
    paths.rows_receipt.write_text(json.dumps(receipt))
    paths.rows_manifest.write_text("rows-manifest")
    return tensors, entries, receipt


def _write_programs_unlock(
    paths: lifecycle.ArtifactPaths,
    source_closure: dict[str, object],
    protected: dict[str, object],
) -> dict[str, object]:
    _touch(paths.programs, b"programs")
    receipt = {
        "schema_version": 1,
        "status": "frozen_programs_before_final",
        "authority": "early_mlp_suffix_transport_v1_programs_unlock",
        "authorized_for_final_scoring": True,
        "rows_receipt": lifecycle.artifact_binding(paths.rows_receipt),
        "programs": lifecycle.artifact_binding(paths.programs),
        "source_commit": source_closure["source_commit"],
        "source_hashes": source_closure["source_hashes"],
        "protected_before": protected,
    }
    paths.programs_receipt.write_text(json.dumps(receipt))
    return receipt


def test_role_loader_enforces_operation_phase_and_requested_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = lifecycle.ArtifactPaths(tmp_path)
    _, entries, _ = _write_row_receipt(paths)
    monkeypatch.setattr(lifecycle, "_validate_rows_receipt", lambda receipt, paths: None)
    lock = tmp_path / "run.lock"

    original_load = torch.load
    seen = []

    def recording_load(path, *args, **kwargs):
        seen.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    with lifecycle.exclusive_run_claim(lock) as nonce:
        _, loaded = lifecycle.load_roles(
            [lifecycle.ROLE_NAMES[0]], operation="training", lock_nonce=nonce,
            paths=paths, lock_path=lock,
        )
        assert set(loaded) == {lifecycle.ROLE_NAMES[0]}
        assert seen == [Path(entries[lifecycle.ROLE_NAMES[0]]["cache_path"])]
        with pytest.raises(RuntimeError, match="operation license"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[1]], operation="training", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )
        with pytest.raises(RuntimeError, match="missing"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[1]], operation="selection", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )
        for path in (paths.fit_ledger, paths.fit_manifest, paths.fit_receipt):
            _touch(path)
        _, loaded = lifecycle.load_roles(
            [lifecycle.ROLE_NAMES[1]], operation="selection", lock_nonce=nonce,
            paths=paths, lock_path=lock,
        )
        assert set(loaded) == {lifecycle.ROLE_NAMES[1]}
        _touch(paths.programs)
        with pytest.raises(RuntimeError, match="ordering"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[0]], operation="training", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )


def test_protected_snapshot_detects_presence_and_content_drift(tmp_path: Path) -> None:
    present = tmp_path / "present"
    absent = tmp_path / "absent"
    present.write_text("one")
    snapshot = lifecycle.protected_snapshot((present, absent))
    lifecycle.require_protected_snapshot((present, absent), snapshot)
    present.write_text("two")
    with pytest.raises(RuntimeError, match="drifted"):
        lifecycle.require_protected_snapshot((present, absent), snapshot)
    present.write_text("one")
    absent.write_text("appeared")
    with pytest.raises(RuntimeError, match="drifted"):
        lifecycle.require_protected_snapshot((present, absent), snapshot)


def test_frozen_inputs_and_neutral_import_contract() -> None:
    lifecycle.verify_frozen_inputs()
    source = Path(lifecycle.__file__).read_text()
    assert "bilin18_joint_removal" not in source
    assert "early_mlp_state_complete_compiler_v21" not in source
    assert "torch.load(" in source  # confined to explicit role loader, never import time


def test_source_closure_rejects_incomplete_set_and_current_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=lifecycle.ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    with pytest.raises(RuntimeError, match="path set"):
        lifecycle.verify_source_closure(current_commit, {})

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    source = repo / "source.py"
    source.write_text("one\n")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    hashes = {"source.py": lifecycle.file_sha256(source)}
    monkeypatch.setattr(lifecycle, "ROOT", repo)
    monkeypatch.setattr(lifecycle, "SOURCE_CLOSURE", (source,))
    lifecycle.verify_source_closure(commit, hashes)
    source.write_text("drift\n")
    with pytest.raises(RuntimeError, match="current source content drifted"):
        lifecycle.verify_source_closure(commit, hashes)


def test_numerical_source_gate_keeps_the_observed_final_executor_fail_closed() -> None:
    closure = set(lifecycle.source_closure_paths())
    assert set(path.resolve() for path in lifecycle.OBSERVED_EXECUTION_CLOSURE) <= closure
    assert set(path.resolve() for path in lifecycle.MAPPED_CONTROL_CLOSURE) <= closure
    assert set(path.resolve() for path in lifecycle.NUMERICAL_STAGE_CLOSURE) <= closure
    assert all(path.is_file() for path in lifecycle.OBSERVED_EXECUTION_CLOSURE)
    assert all(path.is_file() for path in lifecycle.MAPPED_CONTROL_CLOSURE)
    missing = [path.name for path in lifecycle.NUMERICAL_STAGE_CLOSURE if not path.is_file()]
    assert missing == [
        "early_mlp_suffix_transport_v1_final_execution.py",
        "test_early_mlp_suffix_transport_v1_final_execution.py",
    ]
    with pytest.raises(RuntimeError, match="numerical source closure is incomplete") as error:
        lifecycle.require_numerical_source_closure()
    for name in missing:
        assert name in str(error.value)


def test_final_requires_canonical_unlock_attempt_and_owned_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = lifecycle.ArtifactPaths(tmp_path)
    _, entries, _ = _write_row_receipt(paths)
    monkeypatch.setattr(lifecycle, "_validate_rows_receipt", lambda receipt, paths: None)
    source_closure = {"source_commit": "c" * 40, "source_hashes": {"x": "y"}}
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda *_: None)
    protected = {"pin": "same"}
    lock = tmp_path / "run.lock"
    with lifecycle.exclusive_run_claim(lock) as nonce:
        with pytest.raises(RuntimeError, match="canonical programs unlock"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[2]], operation="final", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )
        _write_programs_unlock(paths, source_closure, protected)
        with pytest.raises(RuntimeError, match="attempt must exist"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[2]], operation="final", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )
        attempt = lifecycle.write_final_attempt(
            paths=paths,
            source_closure=source_closure,
            protected_before=protected,
            lock_nonce=nonce,
            lock_path=lock,
        )
        assert attempt["authority"] == "none"
        assert attempt["final_role_loads_before_attempt"] == 0
        _, loaded = lifecycle.load_roles(
            [lifecycle.ROLE_NAMES[2]], operation="final", lock_nonce=nonce,
            paths=paths, lock_path=lock,
        )
        assert torch.equal(loaded[lifecycle.ROLE_NAMES[2]], torch.full((2, 33), 2))
        assert Path(entries[lifecycle.ROLE_NAMES[2]]["cache_path"]).is_file()
        with pytest.raises(RuntimeError, match="exactly once"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[2]], operation="final", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )
        monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 0)
        paths.programs.write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="unlock binding"):
            lifecycle.load_roles(
                [lifecycle.ROLE_NAMES[2]], operation="final", lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )


def test_terminal_publication_is_absent_until_semantic_validator_exists() -> None:
    assert not hasattr(lifecycle, "write_terminal_authority")
