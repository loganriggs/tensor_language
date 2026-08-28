import json
import os
from pathlib import Path
import shutil

import pytest
import torch

import early_mlp_suffix_transport_v1_lifecycle as life
import early_mlp_suffix_transport_v1_row_freezer as freezer
import early_mlp_suffix_transport_v1_rows as identity


def synthetic_texts(candidate_index=0):
    triple = life.candidate_triple(candidate_index)
    final_end = triple.final_skip + triple.final_n
    for index in range(final_end):
        yield f"doc-{index}", str(index) if index >= 43_000 else ""


def encode_synthetic(value):
    if not value:
        return []
    first = int(value) % 50_000
    return [first, *range(1, identity.TOKEN_LENGTH + 1)]


def test_harvest_candidate_obeys_registered_counts_skips_and_order():
    values, records = freezer.harvest_candidate(
        synthetic_texts(), encode_synthetic,
        candidate_index=0, seen_prefixes=set(),
    )
    assert {role: tuple(value.shape) for role, value in values.items()} == {
        "fit": (384, 513), "validation": (192, 513), "final": (192, 513),
    }
    assert records["fit"][0]["dataset_document_index"] == 43_000
    assert records["validation"][0]["dataset_document_index"] == 47_000
    assert records["final"][0]["dataset_document_index"] == 51_000
    assert records["fit"][-1]["dataset_document_index"] == 43_383


def test_harvest_candidate_applies_only_frozen_prefix_filter():
    prefix = (43_000, *range(1, 32))
    _, records = freezer.harvest_candidate(
        synthetic_texts(), encode_synthetic,
        candidate_index=0, seen_prefixes={prefix},
    )
    assert records["fit"][0]["dataset_document_index"] == 43_001


def make_staged(tmp_path, candidate_index=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    values, records = freezer.harvest_candidate(
        synthetic_texts(candidate_index), encode_synthetic,
        candidate_index=candidate_index, seen_prefixes=set(),
    )
    staging = tmp_path / "staging"
    entries = freezer._stage_candidate(staging, values, candidate_index)
    report = identity.adjudicate_candidate(
        candidate_index=candidate_index, rows_by_role=values, records_by_role=records,
        prior=identity.IdentitySets.empty(),
    )
    return values, records, staging, entries, report


def fake_snapshot():
    return {
        "source_closure": {"source_commit": "a" * 40, "source_hashes": {}},
        "gate": {"ordered": True},
        "source": Path("/pinned/source.parquet"),
        "source_identity": {"source": "bound"},
        "prior": identity.IdentitySets.empty(),
        "registry_census": {"registry": "bound"},
    }


def publish_synthetic(
    *, monkeypatch, tmp_path, snapshot=None, snapshot_function=None, **kwargs,
):
    snapshot = fake_snapshot() if snapshot is None else snapshot
    monkeypatch.setattr(
        freezer, "canonical_snapshot",
        (lambda: snapshot) if snapshot_function is None else snapshot_function,
    )
    monkeypatch.setattr(freezer, "replay_candidate_history", lambda **kwargs: None)
    lock = tmp_path / "run.lock"
    with life.exclusive_run_claim(lock) as nonce:
        return freezer.publish_frozen_candidate_locked(
            lock_nonce=nonce, lock_path=lock, selection_snapshot=snapshot, **kwargs,
        )


def test_cache_install_is_create_only_and_preserves_staging(tmp_path):
    _, _, staging, entries, _ = make_staged(tmp_path)
    cache = tmp_path / "cache"
    installed = freezer.install_cache_create_only(staging, entries, cache)
    assert set(installed) == set(life.ROLE_NAMES)
    assert all(Path(entry["cache_path"]).is_file() for entry in installed.values())
    assert all(Path(entry["staged_path"]).is_file() for entry in entries.values())
    with pytest.raises(FileExistsError):
        freezer.install_cache_create_only(staging, entries, cache)


def test_publish_writes_receipt_last_and_never_serializes_rejected_rows(
    tmp_path, monkeypatch,
):
    values0, records0, staging0, entries0, accepted0 = make_staged(tmp_path / "zero", 0)
    shutil.rmtree(staging0)
    values, records, staging, entries, accepted1 = make_staged(tmp_path / "one", 1)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    rejected = json.loads(json.dumps(accepted0))
    rejected["accepted"] = False
    rejected["collision_evidence_count"] = 1
    rejected["prior_collision_counts"]["fit"]["documents"] = 1
    writes = []
    real_write = life.atomic_create_json

    def record_write(value, path):
        writes.append(path.name)
        real_write(value, path)

    monkeypatch.setattr(life, "atomic_create_json", record_write)
    receipt = publish_synthetic(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        paths=paths, staging=staging, staged_entries=entries, rows_by_role=values,
        records_by_role=records,
        history=[rejected, accepted1],
    )
    assert writes == [
        paths.collision_manifest.name, paths.rows_manifest.name, paths.rows_receipt.name,
    ]
    assert receipt["authorized_for_training"] is False
    assert set(receipt["entries"]) == set(life.ROLE_NAMES)
    collision_text = paths.collision_manifest.read_text()
    assert "doc-" not in collision_text and "raw_tokens" not in collision_text
    assert life.artifact_binding(paths.rows_manifest) == receipt["rows_manifest"]


def test_publish_without_rejections_keeps_collision_namespace_absent(tmp_path, monkeypatch):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    receipt = publish_synthetic(
        monkeypatch=monkeypatch, tmp_path=tmp_path,
        paths=paths, staging=staging, staged_entries=entries, rows_by_role=values,
        records_by_role=records, history=[accepted],
    )
    assert not paths.collision_manifest.exists()
    assert receipt["collision_manifest"]["absent"] is True


def test_publish_refuses_any_preexisting_output_or_cache(tmp_path, monkeypatch):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    paths.rows_manifest.write_text("competitor")
    with pytest.raises(RuntimeError, match="already spent"):
        publish_synthetic(
            monkeypatch=monkeypatch, tmp_path=tmp_path,
            paths=paths, staging=staging, staged_entries=entries, rows_by_role=values,
            records_by_role=records, history=[accepted],
        )
    assert paths.rows_manifest.read_text() == "competitor"


def test_staging_uses_exclusive_files(tmp_path):
    values, _, staging, _, _ = make_staged(tmp_path)
    with pytest.raises(FileExistsError):
        freezer._stage_candidate(staging, values, 0)


def test_publish_recomputes_chosen_decision_and_filename(tmp_path, monkeypatch):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    bad = json.loads(json.dumps(accepted))
    bad["role_identity_hashes"]["fit"]["ordered_tensor_raw"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not bind"):
        publish_synthetic(
            monkeypatch=monkeypatch, tmp_path=tmp_path,
            paths=paths, staging=staging, staged_entries=entries,
            rows_by_role=values, records_by_role=records,
            history=[bad],
        )
    entries["fit"]["filename"] = "wrong.pt"
    with pytest.raises(RuntimeError, match="staged row path"):
        publish_synthetic(
            monkeypatch=monkeypatch, tmp_path=tmp_path,
            paths=paths, staging=staging, staged_entries=entries,
            rows_by_role=values, records_by_role=records,
            history=[accepted],
        )


def test_pre_receipt_validation_occurs_after_manifest_before_receipt(tmp_path, monkeypatch):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()

    snapshot = fake_snapshot()
    calls = 0

    def drift_after_manifest():
        nonlocal calls
        calls += 1
        if calls == 2:
            assert paths.rows_manifest.is_file()
            assert not paths.rows_receipt.exists()
            changed = dict(snapshot)
            changed["source_identity"] = {"source": "drifted"}
            return changed
        return snapshot

    with pytest.raises(RuntimeError, match="drifted before receipt"):
        publish_synthetic(
            monkeypatch=monkeypatch, tmp_path=tmp_path, snapshot=snapshot,
            snapshot_function=drift_after_manifest,
            paths=paths, staging=staging, staged_entries=entries,
            rows_by_role=values, records_by_role=records,
            history=[accepted],
        )
    assert not paths.rows_receipt.exists()


@pytest.mark.parametrize("target", ["cache", "manifest"])
def test_last_write_revalidates_newly_installed_artifacts(tmp_path, monkeypatch, target):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()

    snapshot = fake_snapshot()
    calls = 0

    def mutate_on_second_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            if target == "manifest":
                paths.rows_manifest.write_text("changed")
            else:
                installed = next(paths.cache.iterdir())
                with installed.open("ab") as handle:
                    handle.write(b"changed")
        return snapshot

    with pytest.raises(RuntimeError, match="changed before receipt"):
        publish_synthetic(
            monkeypatch=monkeypatch, tmp_path=tmp_path, snapshot=snapshot,
            snapshot_function=mutate_on_second_snapshot,
            paths=paths, staging=staging, staged_entries=entries,
            rows_by_role=values, records_by_role=records,
            history=[accepted],
        )
    assert not paths.rows_receipt.exists()


def test_canonical_publisher_requires_owned_lock(tmp_path, monkeypatch):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    snapshot = fake_snapshot()
    monkeypatch.setattr(freezer, "canonical_snapshot", lambda: snapshot)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    with pytest.raises(RuntimeError, match="claim"):
        freezer.publish_frozen_candidate_locked(
            lock_nonce="not-owned", lock_path=tmp_path / "missing.lock",
            paths=paths, staging=staging, staged_entries=entries,
            rows_by_role=values, records_by_role=records,
            history=[accepted], selection_snapshot=snapshot,
        )


def test_lock_loss_during_final_snapshot_forbids_receipt(tmp_path, monkeypatch):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    snapshot = fake_snapshot()
    lock = tmp_path / "run.lock"
    calls = 0

    def lose_lock_on_final_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            assert paths.rows_manifest.is_file()
            assert not paths.rows_receipt.exists()
            lock.write_text("competitor")
        return snapshot

    monkeypatch.setattr(freezer, "canonical_snapshot", lose_lock_on_final_snapshot)
    monkeypatch.setattr(freezer, "replay_candidate_history", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="claim"):
        with life.exclusive_run_claim(lock) as nonce:
            freezer.publish_frozen_candidate_locked(
                lock_nonce=nonce, lock_path=lock, paths=paths, staging=staging,
                staged_entries=entries, rows_by_role=values,
                records_by_role=records, history=[accepted],
                selection_snapshot=snapshot,
            )
    assert not paths.rows_receipt.exists()


def test_serialized_staging_must_equal_adjudicated_tensor(tmp_path):
    values, _, staging, entries, _ = make_staged(tmp_path)
    path = Path(entries["fit"]["staged_path"])
    forged = values["fit"].clone()
    forged[0, 0] += 1
    torch.save(forged, path)
    entries["fit"]["cache_file_sha256"] = life.file_sha256(path)
    with pytest.raises(RuntimeError, match="serialized staging differs"):
        freezer.validate_staged_payloads(
            staged_entries=entries, rows_by_role=values, candidate_index=0,
        )


def test_canonical_history_replay_binds_harvest_and_every_decision(
    tmp_path, monkeypatch,
):
    values, records, _, _, accepted = make_staged(tmp_path)
    snapshot = fake_snapshot()
    snapshot["source"] = tmp_path / "source.parquet"
    monkeypatch.setattr(freezer, "parquet_texts", lambda source: synthetic_texts())
    monkeypatch.setattr(freezer, "load_dedup_prefixes", lambda: set())

    class Encoding:
        encode_ordinary = staticmethod(encode_synthetic)

    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: Encoding())
    freezer.replay_candidate_history(
        selection_snapshot=snapshot, history=[accepted],
        chosen_rows=values, chosen_records=records,
    )
    changed = json.loads(json.dumps(records))
    changed["fit"][0]["document_id"] = "forged"
    with pytest.raises(RuntimeError, match="differ from canonical harvest"):
        freezer.replay_candidate_history(
            selection_snapshot=snapshot, history=[accepted],
            chosen_rows=values, chosen_records=changed,
        )


def test_all_preregistered_inherited_objects_are_protected():
    expected = {
        "early_mlp_state_complete_compiler_v21_final_authority.json",
        "early_mlp_state_complete_compiler_v21_final_result.pt",
        "early_mlp_state_complete_compiler_v21_programs_receipt.json",
        "early_mlp_state_complete_compiler_v21_programs.pt",
        "joint_early_mlp_pca_composition_authoritative_v3_bases.pt",
        "bilin18_frozen_ship_v2_manifest.json",
        "bilin18_frozen_ship_v2.pt",
    }
    assert {path.name for path in identity.PROTECTED_NONROW_FILES} == expected


def configure_synthetic_downstream_validation(monkeypatch, snapshot):
    monkeypatch.setattr(life, "verify_source_closure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        freezer, "validate_ordered_source",
        lambda: (snapshot["gate"], snapshot["source"]),
    )
    monkeypatch.setattr(
        identity, "load_canonical_prior",
        lambda: (snapshot["prior"], snapshot["registry_census"]),
    )
    monkeypatch.setattr(
        freezer, "_source_identity",
        lambda gate, source, encoding: snapshot["source_identity"],
    )


@pytest.mark.parametrize("mutation,message", [
    ("extra", "schema changed"),
    ("registry", "registry/source identity"),
    ("decision", "decision disagrees"),
    ("provenance", "entry/provenance binding"),
    ("cache_path", "entry/provenance binding"),
])
def test_downstream_receipt_validator_is_fail_closed(
    tmp_path, monkeypatch, mutation, message,
):
    values, records, staging, entries, accepted = make_staged(tmp_path)
    paths = life.ArtifactPaths(root=tmp_path / "out")
    paths.root.mkdir()
    snapshot = fake_snapshot()
    receipt = publish_synthetic(
        monkeypatch=monkeypatch, tmp_path=tmp_path, snapshot=snapshot,
        paths=paths, staging=staging, staged_entries=entries,
        rows_by_role=values, records_by_role=records, history=[accepted],
    )
    configure_synthetic_downstream_validation(monkeypatch, snapshot)
    life._validate_rows_receipt(receipt, paths)
    changed = json.loads(json.dumps(receipt))
    if mutation == "extra":
        changed["redirect"] = "leak"
    elif mutation == "registry":
        changed["registry_census"] = {}
    elif mutation == "decision":
        changed["chosen_decision"]["accepted"] = False
    elif mutation == "provenance":
        role = life.ROLE_NAMES[0]
        changed["document_provenance"]["sets"][role][0]["raw_text"] = "leak"
    else:
        role = life.ROLE_NAMES[0]
        changed["entries"][role]["cache_path"] = str(tmp_path / "redirect.pt")
    with pytest.raises(RuntimeError, match=message):
        life._validate_rows_receipt(changed, paths)
