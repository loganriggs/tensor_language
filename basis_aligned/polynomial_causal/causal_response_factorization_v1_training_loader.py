"""One-use, receipt-bound loader for the factorization v1 training role.

This module may deserialize the FIT bundle only after a separate source-closed
analysis authority is frozen. It exposes only the training adapter output, never the
payload, validation role, model, corpus, or EVAL. The capability is poisoned before
its first fallible read and cannot be retried.
"""

from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

import causal_response_factorization_v1_parent_binding as parent
from causal_response_factorization_v1_fit_adapter import (
    FitTrainingInput,
    training_input_from_fit_payload,
)
import causal_response_tensor_v1_fit_bundle as fit_bundle


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRODUCTION_ANALYSIS_AUTHORITY = (
    HERE / "causal_response_factorization_v1_training_authority.json"
)
PRODUCTION_ANALYSIS_AUDIT = (
    HERE / "causal_response_factorization_v1_training_lifecycle_independent_audit.json"
)
PRODUCTION_TERMINAL_DIRECTORY = (
    HERE / "causal_response_factorization_v1_training_terminal"
)
FIT_PARENT_PATH_FIELDS = (
    "authority", "bundle", "manifest", "receipt", "failure", "terminal", "lock",
)


def _same_physical_path(left: Path, right: Path) -> bool:
    """Compare canonical or existing inode targets."""

    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    try:
        left_stat = left.stat()
        right_stat = right.stat()
    except OSError:
        return False
    return (left_stat.st_dev, left_stat.st_ino) == (right_stat.st_dev, right_stat.st_ino)


def _touches_production_parent(paths: parent.FitParentPaths) -> bool:
    return any(
        _same_physical_path(
            getattr(paths, candidate_field), getattr(parent.PRODUCTION_PATHS, production_field)
        )
        for candidate_field in FIT_PARENT_PATH_FIELDS
        for production_field in FIT_PARENT_PATH_FIELDS
    )


def _opened_record_touches_production(record: Mapping[str, Any]) -> bool:
    """Compare the inode actually read with every protected production role."""

    identity = (record.get("device"), record.get("inode"))
    for field in FIT_PARENT_PATH_FIELDS:
        try:
            observed = getattr(parent.PRODUCTION_PATHS, field).stat()
        except OSError:
            continue
        if identity == (observed.st_dev, observed.st_ino):
            return True
    return False


def _reject_synthetic_opened_production(
    record: Mapping[str, Any], label: str, *, require_production: bool,
) -> None:
    if not require_production and _opened_record_touches_production(record):
        raise RuntimeError(f"synthetic loader opened a production {label} inode")


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_plain_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    before = _artifact_sha256(path)
    raw = path.read_bytes()
    after = _artifact_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"{label} changed during exact read")
    value = json.loads(raw)
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not a plain object")
    return value, before


def _production_authority(
    expected_artifact_sha256: str, parent_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the canonical published authority, audit, sources, and namespace."""
    authority, observed = _stable_plain_json(
        PRODUCTION_ANALYSIS_AUTHORITY, "factor training production authority"
    )
    if observed != expected_artifact_sha256:
        raise RuntimeError("factor training production authority artifact changed")
    _validate_analysis_authority(authority, parent_binding)
    audit, audit_digest = _stable_plain_json(
        PRODUCTION_ANALYSIS_AUDIT, "factor training production audit"
    )
    independent = authority.get("independent_audit")
    closure = authority.get("source_closure")
    if (
        type(independent) is not dict
        or independent.get("path") != str(PRODUCTION_ANALYSIS_AUDIT)
        or independent.get("sha256") != audit_digest
        or audit.get("schema")
        != "causal_response_factorization_v1_training_lifecycle_independent_audit"
        or audit.get("status") != "GO"
        or audit.get("approved") is not True
        or audit.get("outcome_access") is not False
        or audit.get("remaining_execution_blockers") != []
        or type(closure) is not dict
        or closure.get("commit") != audit.get("audited_source_commit")
        or closure.get("paths") != audit.get("audited_source_hashes")
    ):
        raise RuntimeError("factor training production authority provenance changed")
    commit = closure["commit"]
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True,
    ).strip()
    if resolved != commit or subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("factor training production authority source is unpublished")
    for relative, expected in closure["paths"].items():
        if type(relative) is not str or type(expected) is not str:
            raise RuntimeError("factor training production source closure is malformed")
        path = ROOT / relative
        historical = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if (
            historical.returncode != 0
            or hashlib.sha256(historical.stdout).hexdigest() != expected
            or not path.is_file()
            or _artifact_sha256(path) != expected
        ):
            raise RuntimeError(f"factor training production source changed: {relative}")
    closure_body = {"commit": commit, "paths": closure["paths"]}
    if closure.get("sha256") != parent._logical_sha256(closure_body):
        raise RuntimeError("factor training production source closure identity changed")
    expected_outputs = {
        "authority": str(PRODUCTION_ANALYSIS_AUTHORITY),
        "input": str(HERE / "causal_response_factorization_v1_training_input.pt"),
        "manifest": str(HERE / "causal_response_factorization_v1_training_manifest.json"),
        "receipt": str(PRODUCTION_TERMINAL_DIRECTORY / "receipt.json"),
        "failure": str(PRODUCTION_TERMINAL_DIRECTORY / "failure.json"),
        "terminal": str(PRODUCTION_TERMINAL_DIRECTORY / "terminal.json"),
        "terminal_directory": str(PRODUCTION_TERMINAL_DIRECTORY),
        "lock": "/workspace/runs/.causal_response_factorization_v1_training.lock",
    }
    if authority.get("output_paths") != expected_outputs:
        raise RuntimeError("factor training production output namespace changed")
    return authority


def _validate_analysis_authority(
    authority: Mapping[str, Any], parent_binding: Mapping[str, Any]
) -> None:
    if type(authority) is not dict or set(authority) != {
        "schema", "status", "source_closure", "independent_audit",
        "parent_binding_sha256", "protocol", "output_paths",
        "outcome_access_before_authority", "authorized_for_training_input",
        "authorized_for_validation", "authorized_for_eval", "authority_sha256",
    } or authority.get("schema") != (
        "causal_response_factorization_v1_training_authority"
    ) or authority.get("status") != (
        "frozen_before_fit_bundle_tensor_deserialization"
    ) or authority.get("authorized_for_training_input") is not True or (
        authority.get("authorized_for_validation") is not False
        or authority.get("authorized_for_eval") is not False
    ):
        raise RuntimeError("factor training authority schema or role changed")
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if authority["authority_sha256"] != parent._logical_sha256(body):
        raise RuntimeError("factor training authority logical identity does not replay")
    if authority["parent_binding_sha256"] != parent_binding["binding_sha256"]:
        raise RuntimeError("factor training authority binds a different FIT parent")
    if authority["outcome_access_before_authority"] != {
        "fit_bundle_deserialized": False,
        "fit_response_values_read": False,
        "validation_values_read": False,
        "eval_values_read": False,
    }:
        raise RuntimeError("factor training authority outcome boundary changed")
    protocol = authority["protocol"]
    if type(protocol) is not dict or protocol.get("role") != "FIT_TRAINING" or (
        protocol.get("training_documents") != 229
        or protocol.get("validation_documents_exposed") != 0
        or protocol.get("eval_documents_exposed") != 0
    ):
        raise RuntimeError("factor training authority protocol changed")


class OneUseFitTrainingLoader:
    """A source-closed owner must construct one instance and call it exactly once."""

    def __init__(
        self,
        paths: parent.FitParentPaths = parent.PRODUCTION_PATHS,
        *,
        require_production: bool = True,
        train_documents: int = 229,
    ) -> None:
        if not isinstance(paths, parent.FitParentPaths):
            raise TypeError("training loader paths must be an exact FitParentPaths")
        if require_production and (
            train_documents != 229 or any(
                not _same_physical_path(
                    getattr(paths, field), getattr(parent.PRODUCTION_PATHS, field)
                ) for field in FIT_PARENT_PATH_FIELDS
            )
        ):
            raise RuntimeError("production training loader requires canonical paths and 229 documents")
        if not require_production and _touches_production_parent(paths):
            raise RuntimeError("production FIT paths cannot use the synthetic loader surface")
        self._paths = paths
        self._require_production = require_production
        self._train_documents = train_documents
        self._spent = False

    @property
    def spent(self) -> bool:
        return self._spent

    def load_once(
        self,
        *,
        parent_binding: Mapping[str, Any],
        analysis_authority: Mapping[str, Any] | None = None,
        expected_analysis_authority_artifact_sha256: str | None = None,
    ) -> FitTrainingInput:
        if self._spent:
            raise RuntimeError("FIT training loader capability is already spent")
        # Poison before the first file lookup. Failure cannot turn into a second try.
        self._spent = True
        if self._require_production:
            if any(
                not _same_physical_path(
                    getattr(self._paths, field), getattr(parent.PRODUCTION_PATHS, field)
                )
                for field in FIT_PARENT_PATH_FIELDS
            ):
                raise RuntimeError("production training loader paths changed before load")
        elif _touches_production_parent(self._paths):
            raise RuntimeError("synthetic loader paths became production aliases before load")
        replay_parent = parent.fit_parent_binding_without_tensor_load(self._paths)
        if replay_parent != parent_binding:
            raise RuntimeError("FIT parent changed before training tensor access")
        if self._require_production:
            if analysis_authority is not None or not isinstance(
                expected_analysis_authority_artifact_sha256, str
            ):
                raise RuntimeError("production loader requires the canonical authority artifact")
            analysis_authority = _production_authority(
                expected_analysis_authority_artifact_sha256, parent_binding
            )
        elif analysis_authority is None or expected_analysis_authority_artifact_sha256 is not None:
            raise RuntimeError("synthetic loader authority surface is malformed")
        _validate_analysis_authority(analysis_authority, parent_binding)

        if self._require_production:
            if any(
                not _same_physical_path(
                    getattr(self._paths, field), getattr(parent.PRODUCTION_PATHS, field)
                )
                for field in FIT_PARENT_PATH_FIELDS
            ):
                raise RuntimeError("production training loader paths changed during authority replay")
        elif _touches_production_parent(self._paths):
            raise RuntimeError("synthetic loader paths became production aliases during load")

        bundle_record, bundle_raw = parent._stable_record(self._paths.bundle)
        _reject_synthetic_opened_production(
            bundle_record, "bundle", require_production=self._require_production
        )
        if bundle_record["sha256"] != parent_binding["bundle_sha256"] or (
            bundle_record["bytes"] != parent_binding["bundle_bytes"]
        ):
            raise RuntimeError("training loader bundle bytes differ from authority parent")
        payload = torch.load(
            io.BytesIO(bundle_raw), map_location="cpu", weights_only=True
        )
        fit_bundle.validate_fit_bundle_payload(
            payload, require_production=self._require_production
        )

        manifest_record, manifest_raw = parent._stable_record(self._paths.manifest)
        _reject_synthetic_opened_production(
            manifest_record, "manifest", require_production=self._require_production
        )
        if manifest_record["sha256"] != parent_binding["manifest_artifact_sha256"]:
            raise RuntimeError("training loader manifest differs from authority parent")
        manifest = parent._plain_json(manifest_raw, "factor training FIT manifest")
        summary = fit_bundle.fit_bundle_manifest_summary_from_payload(
            payload,
            expected_authority_sha256=parent_binding["authority_logical_sha256"],
            require_production=self._require_production,
        )
        if manifest.get("bundle_summary") != summary:
            raise RuntimeError("training loader manifest summary does not replay")

        receipt_record, receipt_raw = parent._stable_record(self._paths.receipt)
        _reject_synthetic_opened_production(
            receipt_record, "receipt", require_production=self._require_production
        )
        if receipt_record["sha256"] != parent_binding["receipt_sha256"]:
            raise RuntimeError("training loader receipt differs from authority parent")
        receipt = parent._plain_json(receipt_raw, "factor training FIT receipt")
        receipt_payload = receipt.get("payload")
        binding = payload["binding"]
        if type(receipt_payload) is not dict or (
            receipt_payload.get("status") != "complete"
            or receipt_payload.get("authorized_for_eval") is not False
            or receipt_payload.get("model_state_sha256_before")
            != binding["model_state_sha256_before"]
            or receipt_payload.get("model_state_sha256_after")
            != binding["model_state_sha256_after"]
            or receipt_payload.get("outer_forwards")
            != payload["call_ledger"]["outer_forwards"]
            or receipt_payload.get("projection_event_shape") != [2, 49, 124]
            or receipt_payload.get("capture_event_shape") != [6, 124]
        ):
            raise RuntimeError("training loader receipt does not join the bundle")
        checkpoint = receipt_payload.get("checkpoint")
        if type(checkpoint) is not dict or (
            checkpoint.get("config_sha256") != binding["config_sha256"]
            or checkpoint.get("weights_sha256") != binding["weights_sha256"]
        ):
            raise RuntimeError("training loader checkpoint does not join the bundle")

        # Terminal replay after every semantic and deserialization check. The adapter
        # receives a private payload once; no payload or full-role alias is returned.
        if parent.fit_parent_binding_without_tensor_load(self._paths) != parent_binding:
            raise RuntimeError("FIT parent changed during training tensor load")
        result = training_input_from_fit_payload(
            payload,
            parent_binding=parent_binding,
            require_production=self._require_production,
            train_documents=self._train_documents,
        )
        del payload
        del bundle_raw
        return result
