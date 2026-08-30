#!/usr/bin/env python3
"""Audit-gated DESIGN-only fitter that freezes the Rayleigh predictors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mlp2_error_rayleigh_predictor as predictor
import prepare_mlp2_error_rayleigh_v1_rows as row_life
import run_mlp0_c512_mlp2_full512_composition_v1 as base
import run_mlp2_error_rayleigh_v1_collect as collector


RUNNER = Path(__file__).resolve()
TEST = HERE / "test_run_mlp2_error_rayleigh_v1_score_design.py"
AUDIT = HERE / "mlp2_error_rayleigh_v4_design_scorer_independent_audit.json"
AUTHORITY = HERE / "mlp2_error_rayleigh_v4_design_predictor_authority.json"
BUNDLE = HERE / "mlp2_error_rayleigh_v4_design_predictor_bundle.pt"
RECEIPT = HERE / "mlp2_error_rayleigh_v4_design_predictor_receipt.json"
FAILURE = HERE / "mlp2_error_rayleigh_v4_design_predictor_failure.json"
LOCK = Path("/workspace/runs/.mlp2_error_rayleigh_v4_design_predictor.lock")
DESIGN = collector.role_paths("DESIGN")
RECOVERY_AMENDMENT = HERE / "MLP2_ERROR_RAYLEIGH_SCORER_V4_RECOVERY_AMENDMENT.md"
V3_FAILURE = HERE / "mlp2_error_rayleigh_v3_design_predictor_failure.json"
V3_FAILURE_SHA = "d715167e26aec84378d6a48bbcabe8dfd3953cc8d108b959b8b300e88a16c3a6"
V3_ABSENT_PATHS = (
    HERE / "mlp2_error_rayleigh_v3_design_predictor_authority.json",
    HERE / "mlp2_error_rayleigh_v3_design_predictor_bundle.pt",
    HERE / "mlp2_error_rayleigh_v3_design_predictor_receipt.json",
    Path("/workspace/runs/.mlp2_error_rayleigh_v3_design_predictor.lock"),
)
SOURCE_PATHS = tuple(dict.fromkeys((
    *collector.SOURCE_PATHS,
    HERE / "mlp2_error_rayleigh_predictor.py",
    HERE / "test_mlp2_error_rayleigh_predictor.py",
    RUNNER, TEST, RECOVERY_AMENDMENT,
)))


def file_sha256(path: Path) -> str:
    return collector.file_sha256(path)


def source_hashes(commit: str) -> dict[str, str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted DESIGN scorer source: {relative}")
        output[relative] = digest
    return output


def validate_audit(sources: Mapping[str, str]) -> tuple[dict[str, Any], str]:
    raw = AUDIT.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if file_sha256(AUDIT) != digest:
        raise RuntimeError("DESIGN scorer audit changed while reading")
    value = json.loads(raw)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if set(value) != required or value.get("schema") != (
        "mlp2_error_rayleigh_v4_design_scorer_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) \
            or value["tests_passed"] < 1 or not value.get("reviewer"):
        raise RuntimeError("DESIGN scorer audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("DESIGN scorer audit binding changed")
    return value, digest


def audited_source_commit() -> str:
    raw = AUDIT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if file_sha256(AUDIT) != digest:
        raise RuntimeError("DESIGN scorer audit raced source-commit selection")
    value = json.loads(raw)
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str):
        raise RuntimeError("DESIGN scorer audited source commit is absent")
    return commit


def validate_spent_v3_scorer() -> str:
    value, digest = base.stable_json(V3_FAILURE, V3_FAILURE_SHA)
    absences = {str(path): path.exists() for path in V3_ABSENT_PATHS}
    if digest != V3_FAILURE_SHA or value != {
        "artifact_hashes": {},
        "authority_exists": False,
        "authority_observation": {"status": "absent"},
        "design_ledger_may_have_opened": False,
        "error": "CalledProcessError(1, ['git', 'merge-base', '--is-ancestor', "
                 "'25c7681c6321c89e4460195123dd99cc0fcdd9dc', 'origin/main'])",
        "protected_observation": {"status": "not_attempted_preopen_failure"},
        "schema": "mlp2_error_rayleigh_v1_design_predictor_failure",
        "status": "terminal_failure_no_receipt",
    } or any(absences.values()):
        raise RuntimeError("spent Rayleigh v3 scorer chain changed")
    replay, replay_digest = base.stable_json(V3_FAILURE, V3_FAILURE_SHA)
    if replay != value or replay_digest != digest \
            or {str(path): path.exists() for path in V3_ABSENT_PATHS} != absences:
        raise RuntimeError("spent Rayleigh v3 scorer lineage raced validation")
    return digest


def validate_design_receipt(value: Any, ledger_sha: str) -> dict[str, Any]:
    required = {
        "schema", "status", "role", "authority_sha256", "ledger_sha256",
        "runtime_s", "model_responses_opened", "heldout_predictor_was_frozen",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema") != "mlp2_error_rayleigh_v1_collector_receipt" \
            or value.get("status") != "role_measurements_complete_receipt_last" \
            or value.get("role") != "DESIGN" or value.get("ledger_sha256") != ledger_sha \
            or value.get("model_responses_opened") is not True \
            or value.get("heldout_predictor_was_frozen") is not False:
        raise RuntimeError("DESIGN collector receipt changed")
    return value


def validate_design_authority(value: Any, authority_sha: str) -> dict[str, Any]:
    required = {
        "schema", "status", "role", "source_commit", "source_hashes",
        "audit_sha256", "audit_reviewer", "row_receipt_sha256", "row_file_sha256",
        "parent_snapshot", "predictor_unlock_sha256", "programs", "backgrounds",
        "controls", "amplitudes", "control_seed", "scored_slice",
        "attention_capture_sites", "outcome_access",
    }
    if not isinstance(value, dict) or set(value) != required \
            or value.get("schema") != "mlp2_error_rayleigh_v1_collector_authority" \
            or value.get("status") != "frozen_before_role_response_open" \
            or value.get("role") != "DESIGN" \
            or value.get("predictor_unlock_sha256") is not None \
            or value.get("outcome_access") is not False \
            or not isinstance(authority_sha, str) or len(authority_sha) != 64:
        raise RuntimeError("DESIGN collector authority changed")
    collector.protected_snapshot(value)
    return value


def stable_file_sha(path: Path, expected: str | None = None) -> str:
    """Hash a file twice without parsing or deserializing it."""
    first = file_sha256(path)
    if expected is not None and first != expected:
        raise RuntimeError("DESIGN scorer protected file hash changed")
    if not path.is_file() or file_sha256(path) != first:
        raise RuntimeError("DESIGN scorer protected file raced hash")
    return first


def metadata_snapshot(authority: Mapping[str, Any]) -> dict[str, str]:
    """Bind scorer inputs without opening the DESIGN tensor ledger.

    This is the only protected replay allowed before scorer authority publication.
    JSON metadata may be parsed, but DESIGN tensor values must remain unopened.
    """
    if source_hashes(authority["source_commit"]) != authority["source_hashes"]:
        raise RuntimeError("DESIGN scorer sources changed")
    _, audit_sha = validate_audit(authority["source_hashes"])
    receipt, receipt_sha = base.stable_json(DESIGN["receipt"], authority["design_receipt_sha256"])
    ledger_sha = stable_file_sha(DESIGN["ledger"], authority["design_ledger_sha256"])
    validate_design_receipt(receipt, ledger_sha)
    design_authority, design_authority_sha = base.stable_json(
        DESIGN["authority"], authority["design_authority_sha256"],
    )
    if receipt["authority_sha256"] != design_authority_sha:
        raise RuntimeError("DESIGN receipt-to-authority join changed")
    validate_design_authority(design_authority, design_authority_sha)
    if audit_sha != authority["audit_sha256"] or receipt_sha != authority["design_receipt_sha256"]:
        raise RuntimeError("DESIGN scorer protected hashes changed")
    spent_v3_sha = validate_spent_v3_scorer()
    if authority["spent_v3_scorer_failure_sha256"] != spent_v3_sha:
        raise RuntimeError("DESIGN scorer spent-v3 lineage changed")
    return {
        "audit": audit_sha, "receipt": receipt_sha, "ledger": ledger_sha,
        "design_authority": design_authority_sha,
        "spent_v3_scorer_failure": spent_v3_sha,
    }


def protected_snapshot(authority: Mapping[str, Any]) -> dict[str, str]:
    """Post-authority semantic replay, including DESIGN tensor deserialization."""
    metadata = metadata_snapshot(authority)
    ledger, ledger_sha = base.stable_torch(
        DESIGN["ledger"], authority["design_ledger_sha256"],
    )
    design_authority, design_authority_sha = base.stable_json(
        DESIGN["authority"], authority["design_authority_sha256"],
    )
    collector.validate_ledger(
        ledger, design_authority_sha, "DESIGN",
        design_authority["parent_snapshot"]["checkpoint"],
    )
    if ledger_sha != metadata["ledger"]:
        raise RuntimeError("DESIGN scorer ledger raced semantic replay")
    return metadata


def pre_open_guard(claim, authority: Mapping[str, Any], authority_sha: str) -> None:
    """Final cooperative transaction guard before any DESIGN tensor load.

    Keep this ordering exact: replay the already-published scorer authority, reject
    a rival terminal, and then prove ownership of the lock.  No protected tensor
    operation is allowed between this function and ``protected_snapshot``.
    """
    observed, observed_sha = base.stable_json(AUTHORITY, authority_sha)
    if observed != authority or observed_sha != authority_sha:
        raise RuntimeError("DESIGN predictor authority changed at pre-open boundary")
    if any(path.exists() for path in (BUNDLE, RECEIPT, FAILURE)):
        raise RuntimeError("DESIGN predictor terminal appeared at pre-open boundary")
    row_life.base.require_claim(claim, LOCK)


def serialize_fit(value: Mapping[str, Any]) -> dict[str, Any]:
    return predictor.serialize_fit(value)


def validate_bundle(value: Any) -> dict[str, Any]:
    predictor.validate_frozen_bundle(value)
    return value


def artifact_snapshot() -> dict[str, str]:
    return {
        name: file_sha256(path) for name, path in {
            "authority": AUTHORITY, "bundle": BUNDLE,
        }.items() if path.is_file()
    }


def publish_failure(claim, exc: BaseException, authority,
                    protected: Mapping[str, str] | None, opened: bool) -> None:
    def observe_authority() -> dict[str, Any]:
        if not AUTHORITY.is_file():
            return {"status": "absent"}
        try:
            observed, observed_sha = base.stable_json(AUTHORITY)
            return {
                "status": "matches" if authority is not None and observed == authority else "mismatch",
                "sha256": observed_sha,
            }
        except BaseException as observation_error:
            return {"status": "replay_error", "error": repr(observation_error)}

    def observe_protected(authority_status: str) -> dict[str, Any]:
        if not opened:
            return {"status": "not_attempted_preopen_failure"}
        if authority_status != "matches":
            return {"status": "not_attempted_authority_untrusted"}
        if authority is None or not AUTHORITY.is_file():
            return {"status": "not_available"}
        try:
            current = protected_snapshot(authority)
            if protected is None:
                return {"status": "baseline_unavailable_replay_succeeded",
                        "current": dict(current)}
            return {"status": "matches" if current == protected else "mismatch"}
        except BaseException as replay_error:
            return {"status": "replay_error", "error": repr(replay_error)}

    frozen_artifacts = artifact_snapshot()
    authority_observation = observe_authority()
    protected_observation = observe_protected(authority_observation["status"])
    failure = {
        "schema": "mlp2_error_rayleigh_v1_design_predictor_failure",
        "status": "terminal_failure_no_receipt", "error": repr(exc),
        "authority_exists": AUTHORITY.exists(), "design_ledger_may_have_opened": opened,
        "artifact_hashes": frozen_artifacts,
        "authority_observation": authority_observation,
        "protected_observation": protected_observation,
    }

    def failure_guard():
        row_life.base.require_claim(claim, LOCK)
        if RECEIPT.exists() or FAILURE.exists() or artifact_snapshot() != frozen_artifacts:
            raise RuntimeError("DESIGN scorer failure terminal or artifacts raced")
        if AUTHORITY.is_file() and stable_file_sha(
            AUTHORITY, frozen_artifacts.get("authority"),
        ) != frozen_artifacts.get("authority"):
            raise RuntimeError("DESIGN scorer failure authority bytes changed")
        # Protected drift and even authority semantic drift are themselves terminal
        # failure classes.  The observations above preserve them; publication binds
        # the exact bytes present on the failure path rather than requiring them to
        # become healthy again.
        if RECEIPT.exists() or FAILURE.exists() or artifact_snapshot() != frozen_artifacts:
            raise RuntimeError("DESIGN scorer failure terminal raced observation")
        row_life.base.require_claim(claim, LOCK)

    if RECEIPT.exists() or FAILURE.exists():
        return
    base.atomic_json(FAILURE, failure, pre_link_check=failure_guard)


def run() -> None:
    paths = (AUTHORITY, BUNDLE, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in paths):
        raise RuntimeError("DESIGN predictor namespace already exists")
    if not DESIGN["receipt"].is_file() or not DESIGN["ledger"].is_file():
        raise RuntimeError("DESIGN measurements are not receipt-complete")
    claim = row_life.base.acquire_claim(LOCK)
    authority = None; protected = None; opened = False
    try:
        commit = audited_source_commit()
        sources = source_hashes(commit); audit, audit_sha = validate_audit(sources)
        design_receipt_sha = file_sha256(DESIGN["receipt"])
        design_ledger_sha = file_sha256(DESIGN["ledger"])
        design_authority_sha = file_sha256(DESIGN["authority"])
        authority = {
            "schema": "mlp2_error_rayleigh_v1_design_predictor_authority",
            "status": "frozen_before_design_ledger_open", "source_commit": commit,
            "source_hashes": sources, "audit_sha256": audit_sha,
            "audit_reviewer": audit["reviewer"],
            "design_receipt_sha256": design_receipt_sha,
            "design_ledger_sha256": design_ledger_sha,
            "design_authority_sha256": design_authority_sha,
            "ridge_grid": list(predictor.RIDGE_GRID),
            "families": {name: list(features) for name, features in predictor.FAMILIES.items()},
            "heldout_opened": False,
            "spent_v3_scorer_failure_sha256": V3_FAILURE_SHA,
        }
        metadata = metadata_snapshot(authority)

        def authority_guard():
            row_life.base.require_claim(claim, LOCK)
            if any(path.exists() for path in (AUTHORITY, BUNDLE, RECEIPT, FAILURE)):
                raise RuntimeError("DESIGN predictor authority terminal appeared")
            if metadata_snapshot(authority) != metadata:
                raise RuntimeError("DESIGN predictor authority inputs changed")
            if any(path.exists() for path in (AUTHORITY, BUNDLE, RECEIPT, FAILURE)):
                raise RuntimeError("DESIGN predictor authority terminal raced metadata replay")
            row_life.base.require_claim(claim, LOCK)

        base.atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(AUTHORITY)
        observed_authority, observed_authority_sha = base.stable_json(AUTHORITY, authority_sha)
        if observed_authority != authority or observed_authority_sha != authority_sha:
            raise RuntimeError("DESIGN predictor authority changed before ledger access")
        pre_open_guard(claim, authority, authority_sha)
        opened = True
        protected = protected_snapshot(authority)
        ledger, _ = base.stable_torch(DESIGN["ledger"], design_ledger_sha)
        design_receipt, _ = base.stable_json(DESIGN["receipt"], design_receipt_sha)
        replay = collector.validate_ledger(
            ledger, validate_design_receipt(design_receipt, design_ledger_sha)["authority_sha256"],
            "DESIGN",
        )
        fit = predictor.fit_design(replay["features"], replay["finite"])
        bundle = serialize_fit(fit)
        validate_bundle(bundle)

        def bundle_guard():
            row_life.base.require_claim(claim, LOCK)
            if BUNDLE.exists() or RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("DESIGN predictor terminal appeared before bundle")
            if protected_snapshot(authority) != protected:
                raise RuntimeError("DESIGN predictor protected state changed")
            observed, observed_sha = base.stable_json(AUTHORITY, authority_sha)
            if observed != authority or observed_sha != authority_sha:
                raise RuntimeError("DESIGN predictor authority semantic join changed")
            if BUNDLE.exists() or RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("DESIGN predictor terminal raced bundle")
            row_life.base.require_claim(claim, LOCK)

        base.atomic_torch(BUNDLE, bundle, pre_link_check=bundle_guard)
        reloaded, bundle_sha = base.stable_torch(BUNDLE)
        validate_bundle(reloaded)
        receipt = {
            "schema": "mlp2_error_rayleigh_v1_design_predictor_receipt",
            "status": "design_predictor_frozen_receipt_last",
            "design_ledger_sha256": design_ledger_sha,
            "design_receipt_sha256": design_receipt_sha,
            "predictor_authority_sha256": authority_sha,
            "scorer_audit_sha256": audit_sha,
            "predictor_bundle_sha256": bundle_sha, "heldout_unlocked": True,
        }

        def receipt_guard():
            row_life.base.require_claim(claim, LOCK)
            if protected_snapshot(authority) != protected:
                raise RuntimeError("DESIGN predictor protected state changed")
            observed_authority, observed_sha = base.stable_json(AUTHORITY, authority_sha)
            replay_bundle, replay_sha = base.stable_torch(BUNDLE, bundle_sha)
            if observed_authority != authority or observed_sha != authority_sha \
                    or replay_sha != bundle_sha:
                raise RuntimeError("DESIGN predictor receipt semantic join changed")
            validate_bundle(replay_bundle)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("DESIGN predictor terminal raced receipt")
            row_life.base.require_claim(claim, LOCK)

        rendered_receipt = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False)
        print(rendered_receipt)
        base.atomic_json(RECEIPT, receipt, pre_link_check=receipt_guard)
    except BaseException as exc:
        try:
            publish_failure(claim, exc, authority, protected, opened)
        except BaseException:
            pass
        raise
    finally:
        row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    run()
