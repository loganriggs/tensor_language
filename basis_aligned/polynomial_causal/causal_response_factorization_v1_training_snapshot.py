"""Source-closed consumer for the authoritative factorization-v1 training snapshot.

The public production entrypoint has no caller-selected path or authority. It returns
only the sanitized training role after revalidating the complete terminal directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

import causal_response_factorization_v1_training_input as training_input
import causal_response_factorization_v1_training_lifecycle as lifecycle
from causal_response_factorization_v1_fit_adapter import FitTrainingInput


ROOT = Path(__file__).resolve().parents[2]
TERMINAL_DIR = lifecycle.TERMINAL_DIR
SNAPSHOT_NAMES = {
    "authority.json", "audit.json", "training_input.pt", "manifest.json",
}


def _stable_regular_bytes(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"training snapshot member is not regular: {path.name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_stat = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if any(
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) != identity
        for value in (after, path_stat)
    ) or len(raw) != before.st_size:
        raise RuntimeError(f"training snapshot member changed during read: {path.name}")
    return raw, after


def _plain_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not a plain object")
    return value


def _historical_source_closure(commit: str) -> dict[str, Any]:
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True,
    ).strip()
    if resolved != commit or subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("training snapshot source commit is not published ancestry")
    hashes: dict[str, str] = {}
    for path in lifecycle.SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"training snapshot historical source is absent: {relative}")
        hashes[relative] = hashlib.sha256(completed.stdout).hexdigest()
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": lifecycle.logical_sha256(body)}


def _load_snapshot(directory: Path, *, require_production: bool) -> FitTrainingInput:
    directory = directory.resolve(strict=True)
    receipt_path = directory / "receipt.json"
    terminal_path = directory / "terminal.json"
    failure_path = directory / "failure.json"
    if failure_path.exists():
        raise RuntimeError("training snapshot is a failure terminal")
    receipt_raw, receipt_stat = _stable_regular_bytes(receipt_path)
    terminal_raw, terminal_stat = _stable_regular_bytes(terminal_path)
    if receipt_raw != terminal_raw or (
        receipt_stat.st_dev, receipt_stat.st_ino
    ) != (terminal_stat.st_dev, terminal_stat.st_ino):
        raise RuntimeError("training snapshot receipt and terminal are not one exact inode")
    receipt = _plain_json(receipt_raw, "training snapshot receipt")
    if set(receipt) != {
        "schema", "kind", "authority_artifact_sha256", "authority_logical_sha256",
        "payload", "terminal_snapshot",
    } or receipt.get("schema") != "causal_response_factorization_v1_training_terminal" or (
        receipt.get("kind") != "receipt"
    ):
        raise RuntimeError("training snapshot receipt schema changed")
    records = receipt["terminal_snapshot"]
    if type(records) is not dict or set(records) != SNAPSHOT_NAMES:
        raise RuntimeError("training snapshot record set changed")
    expected_census = SNAPSHOT_NAMES | {"receipt.json", "terminal.json"}
    observed_census = {path.name for path in directory.iterdir()}
    if observed_census != expected_census:
        raise RuntimeError("training snapshot directory census changed")

    raw: dict[str, bytes] = {}
    for name in sorted(SNAPSHOT_NAMES):
        record = records[name]
        if type(record) is not dict or set(record) != {
            "path_within_terminal_directory", "sha256", "bytes",
        } or record["path_within_terminal_directory"] != name:
            raise RuntimeError(f"training snapshot record is malformed: {name}")
        member_raw, member_stat = _stable_regular_bytes(directory / name)
        if member_stat.st_size != record["bytes"] or hashlib.sha256(
            member_raw
        ).hexdigest() != record["sha256"]:
            raise RuntimeError(f"training snapshot member hash changed: {name}")
        raw[name] = member_raw

    authority = _plain_json(raw["authority.json"], "training snapshot authority")
    audit = _plain_json(raw["audit.json"], "training snapshot audit")
    manifest = _plain_json(raw["manifest.json"], "training snapshot manifest")
    authority_digest = hashlib.sha256(raw["authority.json"]).hexdigest()
    audit_digest = hashlib.sha256(raw["audit.json"]).hexdigest()
    input_digest = hashlib.sha256(raw["training_input.pt"]).hexdigest()
    manifest_digest = hashlib.sha256(raw["manifest.json"]).hexdigest()
    authority_body = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    if (
        authority_digest != receipt["authority_artifact_sha256"]
        or authority.get("authority_sha256") != receipt["authority_logical_sha256"]
        or authority.get("authority_sha256") != lifecycle.logical_sha256(authority_body)
        or type(authority.get("independent_audit")) is not dict
        or authority["independent_audit"].get("sha256") != audit_digest
        or manifest.get("authority_artifact_sha256") != authority_digest
        or manifest.get("authority_logical_sha256") != authority["authority_sha256"]
        or type(manifest.get("input")) is not dict
        or manifest["input"].get("sha256") != input_digest
        or manifest.get("manifest_sha256") != lifecycle.logical_sha256({
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        })
        or manifest_digest != receipt["payload"].get("manifest_sha256")
        or input_digest != receipt["payload"].get("input_sha256")
    ):
        raise RuntimeError("training snapshot authority/input/manifest join changed")

    value, replay_input_digest = training_input.replay_training_input(
        directory / "training_input.pt",
        expected_analysis_authority_sha256=authority["authority_sha256"],
        expected_artifact_sha256=input_digest,
        require_production=require_production,
    )
    if replay_input_digest != input_digest or (
        authority.get("parent_binding_sha256") != value.artifacts.parent_binding_sha256
        or receipt["payload"].get("fit_parent_binding_sha256")
        != value.artifacts.parent_binding_sha256
    ):
        raise RuntimeError("training snapshot FIT parent binding changed")

    if require_production:
        closure = authority.get("source_closure")
        independent = authority.get("independent_audit")
        if type(closure) is not dict or type(independent) is not dict or (
            audit.get("schema")
            != "causal_response_factorization_v1_training_lifecycle_independent_audit"
            or audit.get("status") != "GO"
            or audit.get("approved") is not True
            or audit.get("outcome_access") is not False
            or audit.get("remaining_execution_blockers") != []
            or audit.get("audited_source_commit") != closure.get("commit")
            or audit.get("audited_source_hashes") != closure.get("paths")
            or _historical_source_closure(closure["commit"]) != closure
        ):
            raise RuntimeError("training snapshot independent authority changed")
        if value.response.shape != (2, 49, 49, 229) or len(value.owner_components) != 6:
            raise RuntimeError("training snapshot production role changed")

    # A second complete census/hash pass is the use-time boundary. Returned tensors
    # are clones, so later file mutation cannot alter the admitted training value.
    if {path.name for path in directory.iterdir()} != expected_census:
        raise RuntimeError("training snapshot census changed during semantic replay")
    for name in SNAPSHOT_NAMES:
        member_raw, _ = _stable_regular_bytes(directory / name)
        if hashlib.sha256(member_raw).hexdigest() != records[name]["sha256"]:
            raise RuntimeError("training snapshot changed during semantic replay")
    return value


def load_production_training_snapshot() -> FitTrainingInput:
    """Load the sole canonical, independently audited 229-document training role."""

    return _load_snapshot(TERMINAL_DIR, require_production=True)


__all__ = ("load_production_training_snapshot",)
