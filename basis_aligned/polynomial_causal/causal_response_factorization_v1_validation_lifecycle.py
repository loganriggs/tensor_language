"""Source-closed, receipt-last transaction scoring the 27 frozen programs on validation.

Amendment 16 transaction. The no-argument production entrypoint: freezes an authority
(published source closure, FIT parent, exact freeze-v2 artifact and its independent
audit, grid terminal, protocol) BEFORE any validation value exists in memory; loads the
27 frozen programs by exact bytes; spends one validation-loader capability; scores every
program on every registered panel; publishes the complete table -> manifest -> one
shared terminal receipt/failure directory. It selects no winner and drops no candidate.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

import torch

import causal_response_factorization_v1_parent_binding as parent
import causal_response_factorization_v1_parent_rebinding as rebinding
import causal_response_factorization_v1_validation_loader as validation_loader
import causal_response_factorization_v1_validation_scorer as scorer
from causal_response_factorization_v1 import ResponseProgram
from causal_response_factorization_v1_training_lifecycle import (
    Claim, _normalized_json_bytes, _publish_json, _rename_directory_noreplace,
    _stable_artifact_bytes, _stage_terminal_snapshot, _validate_staged_terminal,
    _fsync_directory_best_effort, file_sha256, logical_sha256, stable_json,
)
from causal_response_factorization_v1_validation_input import (
    FitValidationInput, PRODUCTION_FREEZE_ARTIFACT_SHA256, validate_candidate_freeze,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AUTHORITY = validation_loader.PRODUCTION_VALIDATION_AUTHORITY
TABLE = validation_loader.PRODUCTION_VALIDATION_TABLE
MANIFEST = validation_loader.PRODUCTION_VALIDATION_MANIFEST
TERMINAL_DIR = validation_loader.PRODUCTION_VALIDATION_TERMINAL_DIRECTORY
LOCK = validation_loader.PRODUCTION_VALIDATION_LOCK
FREEZE = HERE / "causal_response_factorization_v1_candidate_freeze_v2.json"
FREEZE_AUDIT = HERE / "causal_response_factorization_v1_candidate_freeze_v2_independent_audit.json"
GRID_TERMINAL = HERE / "causal_response_factorization_v1_grid_results" / "terminal.json"

SOURCE_PATHS = tuple(ROOT / path for path in (
    "basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_PREREGISTRATION.md",
    *(
        f"basis_aligned/polynomial_causal/CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_{index}.md"
        for index in range(1, 17)
    ),
    "basis_aligned/polynomial_causal/causal_response_factorization_v1.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_fit_adapter.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_parent_binding.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_parent_rebinding.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_training_loader.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_training_lifecycle.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_candidate_freeze_v2.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_validation_input.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_validation_loader.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_validation_scorer.py",
    "basis_aligned/polynomial_causal/causal_response_factorization_v1_validation_lifecycle.py",
    "basis_aligned/polynomial_causal/causal_response_tensor_v1_fit_bundle.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_parent_rebinding.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_validation_input.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_validation_loader.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_validation_scorer.py",
    "basis_aligned/polynomial_causal/test_causal_response_factorization_v1_validation_lifecycle.py",
))
FOCUSED_TEST_FILES = (
    "test_causal_response_factorization_v1_parent_rebinding.py",
    "test_causal_response_factorization_v1_validation_input.py",
    "test_causal_response_factorization_v1_validation_loader.py",
    "test_causal_response_factorization_v1_validation_scorer.py",
    "test_causal_response_factorization_v1_validation_lifecycle.py",
)


def protocol() -> dict[str, Any]:
    return {
        "role": "FIT_INTERNAL_VALIDATION",
        "validation_documents": 114,
        "training_documents_in_codes": 229,
        "training_response_values_exposed": 0,
        "eval_documents_exposed": 0,
        "response_shape": [2, 49, 49, 114],
        "response_dtype": "torch.float64",
        "valid_dtype": "torch.bool",
        "candidate_rank_pairs": 9,
        "candidate_programs": 27,
        "calibration_arm_budgets": list(scorer.CALIBRATION_ARM_BUDGETS),
        "designs": list(scorer.DESIGNS),
        "support_gate": scorer.SUPPORT_GATE,
        "normalization_currency": "training_response_rms from each frozen grid cell receipt",
        "candidates_dropped_after_scoring": 0,
        "winner_selected_inside_scorer": False,
        "artifact_order": ["authority", "table", "manifest", "terminal_receipt"],
        "authorized_for_pareto_analyzer_parent": True,
        "authorized_for_eval": False,
    }


def output_paths() -> dict[str, str]:
    return validation_loader.production_output_paths()


def head_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def source_closure(commit: str) -> dict[str, Any]:
    """Hash every registered source at a published commit; current bytes must match."""

    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True,
    ).strip()
    if resolved != commit or subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("validation source commit is not published ancestry")
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if completed.returncode != 0 or not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"validation source does not replay: {relative}")
        hashes[relative] = digest
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def stable_freeze_inputs() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    freeze, freeze_digest = stable_json(FREEZE)
    audit, audit_digest = stable_json(FREEZE_AUDIT)
    validate_candidate_freeze(
        freeze, audit, freeze_artifact_sha256=freeze_digest,
        audit_artifact_sha256=audit_digest, require_production=True,
    )
    if freeze_digest != PRODUCTION_FREEZE_ARTIFACT_SHA256:
        raise RuntimeError("candidate freeze artifact is not the production freeze")
    return freeze, audit, freeze_digest, audit_digest


def stable_grid_terminal(freeze: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    terminal, digest = stable_json(GRID_TERMINAL)
    if digest != freeze.get("grid_terminal_sha256") or (
        terminal.get("manifest_sha256") != freeze.get("grid_manifest_sha256")
        or terminal.get("schema") != "causal_response_factorization_v1_grid_terminal"
        or terminal.get("status") != "complete_training_only_grid"
        or terminal.get("validation_values_read") is not False
        or terminal.get("eval_values_read") is not False
    ):
        raise RuntimeError("grid terminal does not bind the candidate freeze")
    return terminal, digest


def focused_tests_report() -> dict[str, Any]:
    """Run the focused validation suites; the count is recorded, not trusted as audit."""

    completed = subprocess.run(
        ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *FOCUSED_TEST_FILES],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT}:{HERE}"},
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    passed = 0
    for token in tail.replace(",", " ").split():
        if token.isdigit():
            passed = int(token)
            break
    if completed.returncode != 0 or "passed" not in tail or "failed" in tail or passed < 1:
        raise RuntimeError(f"focused validation suites did not pass cleanly: {tail}")
    return {"files": list(FOCUSED_TEST_FILES), "passed": passed, "summary": tail}


def load_frozen_candidates(
    freeze: Mapping[str, Any], grid_terminal: Mapping[str, Any], *,
    source_groups: torch.Tensor,
) -> list[scorer.FrozenCandidate]:
    """Load each frozen program by exact bytes, bound to the freeze and grid terminal."""

    cells = {
        cell.get("artifact"): cell for cell in grid_terminal.get("cells", [])
        if type(cell) is dict
    }
    candidates: list[scorer.FrozenCandidate] = []
    for record in freeze["candidate_programs"]:
        path = ROOT / record["artifact"]
        raw, observation = _stable_artifact_bytes(path)
        if observation["sha256"] != record["artifact_sha256"] or (
            observation["bytes"] != record["bytes"]
        ):
            raise RuntimeError(f"frozen program bytes changed: {record['artifact']}")
        cell = cells.get(path.name)
        if cell is None or cell.get("artifact_sha256") != record["artifact_sha256"] or (
            cell.get("kind") != "result" or cell.get("healthy") is not True
            or cell.get("validation_values_read") is not False
            or cell.get("eval_values_read") is not False
        ):
            raise RuntimeError(f"frozen program is not a healthy grid result: {path.name}")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if type(payload) is not dict or set(payload) != {
            "schema", "status", "program", "document_codes", "metrics", "receipt",
        } or payload.get("schema") != "causal_response_factorization_v1_grid_cell" or (
            payload.get("status") != "complete_training_only"
        ):
            raise RuntimeError(f"frozen program schema changed: {path.name}")
        program_payload = payload["program"]
        if type(program_payload) is not dict or set(program_payload) != {
            "global_phase", "global_source", "global_target", "private_phase",
            "private_source", "private_target", "source_groups",
        }:
            raise RuntimeError(f"frozen program factor schema changed: {path.name}")
        receipt = payload["receipt"]
        if type(receipt) is not dict or (
            receipt.get("global_rank") != record["global_rank"]
            or receipt.get("private_rank_each_owner") != record["private_rank_each_owner"]
            or receipt.get("seed") != record["seed"]
            or receipt.get("validation_values_read") is not False
            or receipt.get("eval_values_read") is not False
            or receipt.get("training_response_rms") != cell.get("training_response_rms")
            or receipt.get("persistent_values") != record["persistent_values"]
            or receipt.get("per_document_values") != record["per_document_values"]
        ):
            raise RuntimeError(f"frozen program receipt does not bind the freeze: {path.name}")
        candidate = scorer.FrozenCandidate(
            global_rank=int(record["global_rank"]),
            private_rank_each_owner=int(record["private_rank_each_owner"]),
            seed=int(record["seed"]),
            artifact=str(record["artifact"]),
            artifact_sha256=str(record["artifact_sha256"]),
            bytes=int(record["bytes"]),
            persistent_values=int(record["persistent_values"]),
            per_document_values=int(record["per_document_values"]),
            training_response_rms=float(receipt["training_response_rms"]),
            program=ResponseProgram(**program_payload),
            training_codes=payload["document_codes"].clone().contiguous(),
        )
        if not torch.equal(candidate.program.source_groups, source_groups):
            raise RuntimeError(f"frozen program owner topology changed: {path.name}")
        candidates.append(candidate)
        del payload
    return candidates


def acquire_claim() -> Claim:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(LOCK, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("validation lifecycle is already locked") from error
    os.write(descriptor, (nonce + "\n").encode())
    os.fsync(descriptor)
    value = os.fstat(descriptor)
    return Claim(descriptor, value.st_dev, value.st_ino, nonce)


def require_claim(claim: Claim) -> None:
    current = os.fstat(claim.descriptor)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(LOCK, flags)
    try:
        observed = os.fstat(descriptor)
        raw = os.read(descriptor, 4096)
        overflow = os.read(descriptor, 1)
    finally:
        os.close(descriptor)
    path_stat = LOCK.stat(follow_symlinks=False)
    identity = (claim.device, claim.inode)
    if any((value.st_dev, value.st_ino) != identity for value in (
        current, observed, path_stat
    )) or overflow or raw != (claim.nonce + "\n").encode():
        raise RuntimeError("validation owner claim changed")


def release_claim(claim: Claim) -> None:
    try:
        try:
            require_claim(claim)
        except Exception:
            pass
        else:
            try:
                LOCK.unlink()
            except OSError:
                pass
    finally:
        try:
            os.close(claim.descriptor)
        except OSError:
            pass


def _publish_terminal_pair(
    value: Mapping[str, Any], *, kind: str, claim: Claim, final_guard,
    snapshot_sources: Mapping[str, tuple[Path, str | None]] | None = None,
) -> str:
    """Atomically publish terminal plus receipt/failure as one directory rename."""

    if kind not in ("receipt", "failure"):
        raise ValueError("validation terminal kind is malformed")
    if "terminal_snapshot" in value:
        raise RuntimeError("terminal snapshot is owned by the publisher")
    staging = Path(tempfile.mkdtemp(
        prefix=f".{TERMINAL_DIR.name}.", dir=TERMINAL_DIR.parent
    ))
    try:
        snapshot_paths, snapshot_records = _stage_terminal_snapshot(
            staging, {} if snapshot_sources is None else snapshot_sources
        )
        normalized = json.loads(json.dumps(
            {**value, "terminal_snapshot": snapshot_records},
            sort_keys=True, allow_nan=False,
        ))
        raw = _normalized_json_bytes(normalized)
        digest = hashlib.sha256(raw).hexdigest()
        payload = staging / f"{kind}.json"
        terminal = staging / "terminal.json"
        descriptor = os.open(payload, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        os.link(payload, terminal)
        if file_sha256(payload) != digest or file_sha256(terminal) != digest or (
            os.stat(payload).st_ino != os.stat(terminal).st_ino
        ):
            raise RuntimeError("staged validation terminal pair does not replay")
        _fsync_directory_best_effort(staging)
        require_claim(claim)
        final_guard(snapshot_paths, snapshot_records)
        _validate_staged_terminal(
            staging, kind=kind, snapshot_records=snapshot_records, terminal_sha256=digest,
        )
        _rename_directory_noreplace(staging, TERMINAL_DIR)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return digest


def _freeze_authority(claim: Claim, attempt: dict[str, Any]) -> tuple[dict[str, Any], str]:
    commit = head_commit()
    closure = source_closure(commit)
    tests = focused_tests_report()
    freeze, freeze_audit, freeze_digest, audit_digest = stable_freeze_inputs()
    grid_terminal, grid_digest = stable_grid_terminal(freeze)
    fit_parent = rebinding.fit_parent_binding_by_content_identity()
    physical_identity = rebinding.physical_identity_deviation()
    body = {
        "schema": validation_loader.AUTHORITY_SCHEMA,
        "status": validation_loader.AUTHORITY_STATUS,
        "source_closure": closure,
        "self_review": {
            "independent_audit": None,
            "independent_audit_note": (
                "No independent auditor was available on this instance; the focused "
                "suites below were run by the owner immediately before authority freeze. "
                "This table is therefore self-reviewed and must be labelled as such "
                "wherever it is cited."
            ),
            "focused_tests": tests,
            "fit_parent_physical_identity_deviation": physical_identity,
        },
        "parent_binding_sha256": fit_parent["binding_sha256"],
        "candidate_freeze": {
            "path": str(FREEZE), "artifact_sha256": freeze_digest,
            "manifest_sha256": freeze["manifest_sha256"],
            "audit_path": str(FREEZE_AUDIT), "audit_artifact_sha256": audit_digest,
            "candidate_programs": freeze["candidate_program_count"],
            "candidate_rank_pairs": freeze["candidate_rank_pair_count"],
        },
        "grid_terminal": {"path": str(GRID_TERMINAL), "artifact_sha256": grid_digest},
        "protocol": protocol(),
        "output_paths": output_paths(),
        "outcome_access_before_authority": dict(validation_loader.OUTCOME_BOUNDARY),
        "authorized_for_validation_scoring": True,
        "authorized_for_candidate_selection": False,
        "authorized_for_eval": False,
    }
    authority = {**body, "authority_sha256": logical_sha256(body)}
    attempt.update({
        "authority": authority,
        "authority_artifact_sha256": hashlib.sha256(
            _normalized_json_bytes(authority)
        ).hexdigest(),
        "fit_parent": fit_parent,
        "freeze": freeze, "freeze_audit": freeze_audit,
        "freeze_digest": freeze_digest, "audit_digest": audit_digest,
        "grid_terminal": grid_terminal, "grid_digest": grid_digest,
        "source_closure": closure,
    })

    def guard() -> None:
        if source_closure(commit) != closure or (
            stable_freeze_inputs() != (freeze, freeze_audit, freeze_digest, audit_digest)
            or stable_grid_terminal(freeze) != (grid_terminal, grid_digest)
            or rebinding.fit_parent_binding_by_content_identity() != fit_parent
        ):
            raise RuntimeError("validation authority protected state changed")
        require_claim(claim)

    digest = _publish_json(authority, AUTHORITY, guard)
    return authority, digest


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _validation_binding(value: FitValidationInput) -> dict[str, Any]:
    body = {
        "response_sha256": _tensor_sha256(value.response),
        "valid_sha256": _tensor_sha256(value.valid),
        "document_ids_sha256": _tensor_sha256(value.document_ids),
        "original_document_indices_sha256": _tensor_sha256(value.original_document_indices),
        "source_groups_sha256": _tensor_sha256(value.source_groups),
        "shape": list(value.response.shape),
        "owner_components": list(value.owner_components),
        "phases": list(value.phases),
        "artifact_binding_bundle_sha256": value.artifacts.bundle_sha256,
        "candidate_freeze_artifact_sha256": value.candidate_freeze.artifact_sha256,
    }
    return {**body, "sha256": logical_sha256(body)}


def execute_validation_v1() -> str:
    """Score the frozen library on validation; return the terminal digest only."""

    if any(path.exists() for path in (AUTHORITY, TABLE, MANIFEST, TERMINAL_DIR, LOCK)):
        raise RuntimeError("validation output namespace is spent")
    claim = acquire_claim()
    attempt: dict[str, Any] = {}
    try:
        authority, authority_digest = _freeze_authority(claim, attempt)
        capability = validation_loader.OneUseFitValidationLoader()
        validation = capability.load_once(
            parent_binding=attempt["fit_parent"],
            candidate_freeze=attempt["freeze"],
            candidate_freeze_audit=attempt["freeze_audit"],
            candidate_freeze_artifact_sha256=attempt["freeze_digest"],
            candidate_freeze_audit_artifact_sha256=attempt["audit_digest"],
            expected_validation_authority_artifact_sha256=authority_digest,
        )
        candidates = load_frozen_candidates(
            attempt["freeze"], attempt["grid_terminal"],
            source_groups=validation.source_groups,
        )
        binding = _validation_binding(validation)
        table_body = scorer.score_library(candidates, validation, attempt["freeze"])
        table_body = {
            **table_body,
            "authority_artifact_sha256": authority_digest,
            "authority_logical_sha256": authority["authority_sha256"],
            "validation_binding": binding,
            "candidate_freeze_artifact_sha256": attempt["freeze_digest"],
            "grid_terminal_sha256": attempt["grid_digest"],
            "source_closure_sha256": authority["source_closure"]["sha256"],
        }
        table = {**table_body, "table_sha256": logical_sha256(table_body)}

        def table_guard() -> None:
            replay, replay_digest = stable_json(AUTHORITY)
            if replay != authority or replay_digest != authority_digest or (
                source_closure(authority["source_closure"]["commit"])
                != authority["source_closure"]
                or stable_freeze_inputs()[2] != attempt["freeze_digest"]
                or rebinding.fit_parent_binding_by_content_identity() != attempt["fit_parent"]
            ):
                raise RuntimeError("validation table protected state changed")
            require_claim(claim)

        table_digest = _publish_json(table, TABLE, table_guard)
        manifest_body = {
            "schema": "causal_response_factorization_v1_validation_manifest",
            "status": "complete_all_candidates_table_semantically_replayed",
            "authority_artifact_sha256": authority_digest,
            "authority_logical_sha256": authority["authority_sha256"],
            "fit_parent_binding_sha256": attempt["fit_parent"]["binding_sha256"],
            "candidate_freeze_artifact_sha256": attempt["freeze_digest"],
            "table": {
                "path": str(TABLE), "sha256": table_digest, "bytes": TABLE.stat().st_size,
                "table_sha256": table["table_sha256"],
                "candidate_count": table["candidate_count"],
            },
            "validation_binding_sha256": binding["sha256"],
            "protocol": protocol(),
            "candidate_selected": False,
            "authorized_for_eval": False,
        }
        manifest = {**manifest_body, "manifest_sha256": logical_sha256(manifest_body)}

        def manifest_guard() -> None:
            authority_replay, authority_observed = stable_json(AUTHORITY)
            table_replay, table_observed = stable_json(TABLE)
            if (
                authority_replay != authority or authority_observed != authority_digest
                or table_replay != table or table_observed != table_digest
                or source_closure(authority["source_closure"]["commit"])
                != authority["source_closure"]
                or rebinding.fit_parent_binding_by_content_identity() != attempt["fit_parent"]
            ):
                raise RuntimeError("validation manifest protected state changed")
            require_claim(claim)

        manifest_digest = _publish_json(manifest, MANIFEST, manifest_guard)
        receipt = {
            "schema": "causal_response_factorization_v1_validation_terminal",
            "kind": "receipt",
            "authority_artifact_sha256": authority_digest,
            "authority_logical_sha256": authority["authority_sha256"],
            "payload": {
                "status": "complete_all_candidates_receipt_last",
                "fit_parent_binding_sha256": attempt["fit_parent"]["binding_sha256"],
                "candidate_freeze_artifact_sha256": attempt["freeze_digest"],
                "table_sha256": table_digest,
                "manifest_sha256": manifest_digest,
                "validation_documents": int(validation.response.shape[-1]),
                "candidate_programs_scored": table["candidate_count"],
                "candidates_dropped_after_scoring": 0,
                "candidate_selected": False,
                "validation_values_read": True,
                "eval_values_read": False,
                "authorized_for_pareto_analyzer_parent": True,
                "authorized_for_eval": False,
            },
        }

        def receipt_guard(snapshot_paths, snapshot_records) -> None:
            authority_replay, authority_observed = stable_json(snapshot_paths["authority.json"])
            table_replay, table_observed = stable_json(snapshot_paths["table.json"])
            manifest_replay, manifest_observed = stable_json(snapshot_paths["manifest.json"])
            freeze_observed = file_sha256(snapshot_paths["candidate_freeze_v2.json"])
            audit_observed = file_sha256(snapshot_paths["candidate_freeze_v2_audit.json"])
            if (
                authority_replay != authority or authority_observed != authority_digest
                or table_replay != table or table_observed != table_digest
                or manifest_replay != manifest or manifest_observed != manifest_digest
                or freeze_observed != attempt["freeze_digest"]
                or audit_observed != attempt["audit_digest"]
                or set(snapshot_records) != {
                    "authority.json", "table.json", "manifest.json",
                    "candidate_freeze_v2.json", "candidate_freeze_v2_audit.json",
                }
            ):
                raise RuntimeError("validation terminal snapshot does not replay")

        return _publish_terminal_pair(
            receipt, kind="receipt", claim=claim, final_guard=receipt_guard,
            snapshot_sources={
                "authority.json": (AUTHORITY, authority_digest),
                "table.json": (TABLE, table_digest),
                "manifest.json": (MANIFEST, manifest_digest),
                "candidate_freeze_v2.json": (FREEZE, attempt["freeze_digest"]),
                "candidate_freeze_v2_audit.json": (FREEZE_AUDIT, attempt["audit_digest"]),
            },
        )
    except Exception as error:
        attempt_authority = attempt.get("authority")
        attempt_digest = attempt.get("authority_artifact_sha256")
        if (
            type(attempt_authority) is dict and isinstance(attempt_digest, str)
            and not TERMINAL_DIR.exists()
        ):
            failure = {
                "schema": "causal_response_factorization_v1_validation_terminal",
                "kind": "failure",
                "attempt_authority_artifact_sha256": attempt_digest,
                "attempt_authority_logical_sha256": attempt_authority["authority_sha256"],
                "payload": {
                    "status": "failed_no_validation_receipt",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "candidate_selected": False,
                    "authorized_for_eval": False,
                },
            }
            failure_sources: dict[str, tuple[Path, str | None]] = {}
            for name, path in (
                ("authority.json", AUTHORITY), ("table.json", TABLE),
                ("manifest.json", MANIFEST),
            ):
                if path.is_file():
                    failure_sources[name] = (path, None)

            def failure_guard(snapshot_paths, snapshot_records) -> None:
                if set(snapshot_paths) != set(failure_sources) or (
                    set(snapshot_records) != set(failure_sources)
                ):
                    raise RuntimeError("validation failure snapshot does not replay")
                for name, path in snapshot_paths.items():
                    if file_sha256(path) != snapshot_records[name]["sha256"]:
                        raise RuntimeError("validation failure snapshot changed")

            _publish_terminal_pair(
                failure, kind="failure", claim=claim, final_guard=failure_guard,
                snapshot_sources=failure_sources,
            )
        raise
    finally:
        release_claim(claim)


def main() -> None:
    print(execute_validation_v1())


if __name__ == "__main__":
    main()
