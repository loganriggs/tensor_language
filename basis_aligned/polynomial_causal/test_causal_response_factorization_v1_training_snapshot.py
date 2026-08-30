import hashlib
import json
import os

import pytest

import causal_response_factorization_v1_training_input as training_input
import causal_response_factorization_v1_training_lifecycle as lifecycle
import causal_response_factorization_v1_training_snapshot as snapshot
from test_causal_response_factorization_v1_fit_adapter import _analysis_input


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _synthetic_snapshot(tmp_path):
    directory = tmp_path / "terminal"
    directory.mkdir()
    audit = {"schema": "synthetic_independent_audit", "outcome_access": False}
    _write_json(directory / "audit.json", audit)
    audit_digest = lifecycle.file_sha256(directory / "audit.json")
    _, value = _analysis_input()
    authority_body = {
        "schema": "causal_response_factorization_v1_training_authority",
        "independent_audit": {"sha256": audit_digest},
        "parent_binding_sha256": value.artifacts.parent_binding_sha256,
    }
    authority = {
        **authority_body, "authority_sha256": lifecycle.logical_sha256(authority_body)
    }
    _write_json(directory / "authority.json", authority)
    authority_digest = lifecycle.file_sha256(directory / "authority.json")
    payload = training_input.build_training_input_payload(
        value, analysis_authority_sha256=authority["authority_sha256"]
    )
    input_digest = training_input.publish_training_input(
        directory / "training_input.pt", payload,
        expected_analysis_authority_sha256=authority["authority_sha256"],
        require_production=False,
    )
    manifest_body = {
        "authority_artifact_sha256": authority_digest,
        "authority_logical_sha256": authority["authority_sha256"],
        "input": {"sha256": input_digest},
    }
    manifest = {
        **manifest_body, "manifest_sha256": lifecycle.logical_sha256(manifest_body)
    }
    _write_json(directory / "manifest.json", manifest)
    manifest_digest = lifecycle.file_sha256(directory / "manifest.json")
    records = {}
    for name in snapshot.SNAPSHOT_NAMES:
        path = directory / name
        records[name] = {
            "path_within_terminal_directory": name,
            "sha256": lifecycle.file_sha256(path),
            "bytes": path.stat().st_size,
        }
    receipt = {
        "schema": "causal_response_factorization_v1_training_terminal",
        "kind": "receipt",
        "authority_artifact_sha256": authority_digest,
        "authority_logical_sha256": authority["authority_sha256"],
        "payload": {
            "input_sha256": input_digest,
            "manifest_sha256": manifest_digest,
            "fit_parent_binding_sha256": value.artifacts.parent_binding_sha256,
        },
        "terminal_snapshot": records,
    }
    _write_json(directory / "receipt.json", receipt)
    os.link(directory / "receipt.json", directory / "terminal.json")
    return directory


def test_snapshot_consumer_returns_only_receipt_bound_training_role(tmp_path):
    directory = _synthetic_snapshot(tmp_path)
    value = snapshot._load_snapshot(directory, require_production=False)
    assert value.response.shape == (2, 4, 4, 3)
    assert not hasattr(value, "validation_response")
    assert not hasattr(value, "eval_response")


@pytest.mark.parametrize("attack", ["extra", "missing", "member", "record"])
def test_snapshot_consumer_rejects_census_member_and_record_tampering(tmp_path, attack):
    directory = _synthetic_snapshot(tmp_path)
    if attack == "extra":
        (directory / "unrecorded.bin").write_bytes(b"extra")
    elif attack == "missing":
        (directory / "manifest.json").unlink()
    elif attack == "member":
        (directory / "authority.json").write_bytes(b'{"mutated":true}\n')
    else:
        receipt = json.loads((directory / "receipt.json").read_text())
        receipt["terminal_snapshot"]["authority.json"]["sha256"] = "f" * 64
        _write_json(directory / "receipt.json", receipt)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        snapshot._load_snapshot(directory, require_production=False)


def test_snapshot_consumer_rejects_receipt_terminal_inode_split(tmp_path):
    directory = _synthetic_snapshot(tmp_path)
    raw = (directory / "terminal.json").read_bytes()
    (directory / "terminal.json").unlink()
    (directory / "terminal.json").write_bytes(raw)
    with pytest.raises(RuntimeError, match="one exact inode"):
        snapshot._load_snapshot(directory, require_production=False)


def test_snapshot_public_api_has_no_path_or_authority_argument():
    assert snapshot.load_production_training_snapshot.__code__.co_argcount == 0
    assert snapshot.__all__ == ("load_production_training_snapshot",)
