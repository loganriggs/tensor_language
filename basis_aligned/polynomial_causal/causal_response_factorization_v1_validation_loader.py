"""One-use, source-closed loader exposing only the 114-document validation role.

Mirror of the training loader (Amendment 16): the owner constructs one instance and
calls it exactly once after the validation authority is published. The loader replays
the FIT parent, the published validation authority and its source closure, the exact
freeze artifact and its independent audit, then hands the private FIT payload to the
pure reducer and returns only cloned validation tensors. No training response, EVAL
value, model, corpus, or candidate tensor is reachable from this module.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

import causal_response_factorization_v1_parent_binding as parent
import causal_response_factorization_v1_parent_rebinding as rebinding
import causal_response_tensor_v1_fit_bundle as fit_bundle
from causal_response_factorization_v1_training_loader import (
    FIT_PARENT_PATH_FIELDS, _artifact_sha256, _production_parent_identities,
    _reject_synthetic_opened_production, _same_physical_path, _stable_plain_json,
    _touches_production_parent,
)
from causal_response_factorization_v1_validation_input import (
    FitValidationInput, PRODUCTION_FREEZE_ARTIFACT_SHA256, validation_input_from_fit_payload,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PRODUCTION_VALIDATION_AUTHORITY = (
    HERE / "causal_response_factorization_v1_validation_authority.json"
)
PRODUCTION_VALIDATION_TABLE = HERE / "causal_response_factorization_v1_validation_table.json"
PRODUCTION_VALIDATION_MANIFEST = (
    HERE / "causal_response_factorization_v1_validation_manifest.json"
)
PRODUCTION_VALIDATION_TERMINAL_DIRECTORY = (
    HERE / "causal_response_factorization_v1_validation_terminal"
)
PRODUCTION_VALIDATION_LOCK = Path(
    "/workspace/runs/.causal_response_factorization_v1_validation.lock"
)
AUTHORITY_SCHEMA = "causal_response_factorization_v1_validation_authority"
AUTHORITY_STATUS = "frozen_before_validation_response_exposure"
AUTHORITY_KEYS = {
    "schema", "status", "source_closure", "self_review", "parent_binding_sha256",
    "candidate_freeze", "grid_terminal", "protocol", "output_paths",
    "outcome_access_before_authority", "authorized_for_validation_scoring",
    "authorized_for_candidate_selection", "authorized_for_eval", "authority_sha256",
}
OUTCOME_BOUNDARY = {
    "fit_bundle_deserialized": False,
    "training_response_values_read": False,
    "validation_values_read": False,
    "eval_values_read": False,
    "candidate_tensors_deserialized": False,
}


def production_output_paths() -> dict[str, str]:
    return {
        "authority": str(PRODUCTION_VALIDATION_AUTHORITY),
        "table": str(PRODUCTION_VALIDATION_TABLE),
        "manifest": str(PRODUCTION_VALIDATION_MANIFEST),
        "receipt": str(PRODUCTION_VALIDATION_TERMINAL_DIRECTORY / "receipt.json"),
        "failure": str(PRODUCTION_VALIDATION_TERMINAL_DIRECTORY / "failure.json"),
        "terminal": str(PRODUCTION_VALIDATION_TERMINAL_DIRECTORY / "terminal.json"),
        "terminal_directory": str(PRODUCTION_VALIDATION_TERMINAL_DIRECTORY),
        "lock": str(PRODUCTION_VALIDATION_LOCK),
    }


def validate_validation_authority(
    authority: Mapping[str, Any], parent_binding: Mapping[str, Any], *,
    candidate_freeze_artifact_sha256: str,
) -> None:
    if type(authority) is not dict or set(authority) != AUTHORITY_KEYS or (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("authorized_for_validation_scoring") is not True
        or authority.get("authorized_for_candidate_selection") is not False
        or authority.get("authorized_for_eval") is not False
    ):
        raise RuntimeError("validation authority schema or role changed")
    body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if authority["authority_sha256"] != parent._logical_sha256(body):
        raise RuntimeError("validation authority logical identity does not replay")
    if authority["parent_binding_sha256"] != parent_binding["binding_sha256"]:
        raise RuntimeError("validation authority binds a different FIT parent")
    if authority["outcome_access_before_authority"] != OUTCOME_BOUNDARY:
        raise RuntimeError("validation authority outcome boundary changed")
    freeze = authority["candidate_freeze"]
    if type(freeze) is not dict or (
        freeze.get("artifact_sha256") != candidate_freeze_artifact_sha256
    ):
        raise RuntimeError("validation authority binds a different candidate freeze")
    protocol = authority["protocol"]
    if type(protocol) is not dict or protocol.get("role") != "FIT_INTERNAL_VALIDATION" or (
        protocol.get("validation_documents") != 114
        or protocol.get("training_response_values_exposed") != 0
        or protocol.get("eval_documents_exposed") != 0
        or protocol.get("candidate_programs") != 27
        or protocol.get("candidates_dropped_after_scoring") != 0
        or protocol.get("winner_selected_inside_scorer") is not False
    ):
        raise RuntimeError("validation authority protocol changed")


def _production_validation_authority(
    expected_artifact_sha256: str, parent_binding: Mapping[str, Any], *,
    candidate_freeze_artifact_sha256: str,
) -> dict[str, Any]:
    """Replay the canonical published validation authority, sources, and namespace."""

    authority, observed = _stable_plain_json(
        PRODUCTION_VALIDATION_AUTHORITY, "validation production authority"
    )
    if observed != expected_artifact_sha256:
        raise RuntimeError("validation production authority artifact changed")
    validate_validation_authority(
        authority, parent_binding,
        candidate_freeze_artifact_sha256=candidate_freeze_artifact_sha256,
    )
    if candidate_freeze_artifact_sha256 != PRODUCTION_FREEZE_ARTIFACT_SHA256:
        raise RuntimeError("validation production authority binds a non-production freeze")
    closure = authority.get("source_closure")
    if type(closure) is not dict or type(closure.get("paths")) is not dict:
        raise RuntimeError("validation production source closure is malformed")
    commit = closure.get("commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise RuntimeError("validation production source commit is malformed")
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=ROOT, text=True,
    ).strip()
    if resolved != commit or subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("validation production authority source is unpublished")
    for relative, expected in closure["paths"].items():
        if type(relative) is not str or type(expected) is not str:
            raise RuntimeError("validation production source closure is malformed")
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
            raise RuntimeError(f"validation production source changed: {relative}")
    closure_body = {"commit": commit, "paths": closure["paths"]}
    if closure.get("sha256") != parent._logical_sha256(closure_body):
        raise RuntimeError("validation production source closure identity changed")
    if authority.get("output_paths") != production_output_paths():
        raise RuntimeError("validation production output namespace changed")
    return authority


class OneUseFitValidationLoader:
    """A source-closed owner must construct one instance and call it exactly once."""

    def __init__(
        self,
        paths: parent.FitParentPaths = parent.PRODUCTION_PATHS,
        *,
        require_production: bool = True,
        train_documents: int = 229,
    ) -> None:
        if not isinstance(paths, parent.FitParentPaths):
            raise TypeError("validation loader paths must be an exact FitParentPaths")
        if require_production and (
            train_documents != 229 or any(
                not _same_physical_path(
                    getattr(paths, field), getattr(parent.PRODUCTION_PATHS, field)
                ) for field in FIT_PARENT_PATH_FIELDS
            )
        ):
            raise RuntimeError(
                "production validation loader requires canonical paths and the 229/114 split"
            )
        if not require_production and _touches_production_parent(paths):
            raise RuntimeError("production FIT paths cannot use the synthetic loader surface")
        self._paths = paths
        self._require_production = require_production
        self._train_documents = train_documents
        self._spent = False

    @property
    def spent(self) -> bool:
        return self._spent

    def _require_paths(self, stage: str) -> None:
        if self._require_production:
            if any(
                not _same_physical_path(
                    getattr(self._paths, field), getattr(parent.PRODUCTION_PATHS, field)
                ) for field in FIT_PARENT_PATH_FIELDS
            ):
                raise RuntimeError(f"production validation loader paths changed {stage}")
        elif _touches_production_parent(self._paths):
            raise RuntimeError(f"synthetic loader paths became production aliases {stage}")

    def load_once(
        self,
        *,
        parent_binding: Mapping[str, Any],
        candidate_freeze: Mapping[str, Any],
        candidate_freeze_audit: Mapping[str, Any],
        candidate_freeze_artifact_sha256: str,
        candidate_freeze_audit_artifact_sha256: str,
        validation_authority: Mapping[str, Any] | None = None,
        expected_validation_authority_artifact_sha256: str | None = None,
    ) -> FitValidationInput:
        if self._spent:
            raise RuntimeError("FIT validation loader capability is already spent")
        # Poison before the first file lookup. Failure cannot turn into a second try.
        self._spent = True
        self._require_paths("before load")
        replay_parent = rebinding.fit_parent_binding_by_content_identity(self._paths)
        if replay_parent != parent_binding:
            raise RuntimeError("FIT parent changed before validation tensor access")
        if self._require_production:
            if validation_authority is not None or not isinstance(
                expected_validation_authority_artifact_sha256, str
            ):
                raise RuntimeError("production loader requires the canonical authority artifact")
            validation_authority = _production_validation_authority(
                expected_validation_authority_artifact_sha256, parent_binding,
                candidate_freeze_artifact_sha256=candidate_freeze_artifact_sha256,
            )
        elif validation_authority is None or (
            expected_validation_authority_artifact_sha256 is not None
        ):
            raise RuntimeError("synthetic loader authority surface is malformed")
        validate_validation_authority(
            validation_authority, parent_binding,
            candidate_freeze_artifact_sha256=candidate_freeze_artifact_sha256,
        )
        self._require_paths("during authority replay")

        forbidden = frozenset() if self._require_production else _production_parent_identities()
        bundle_record, bundle_raw = parent._stable_record(
            self._paths.bundle, forbidden_identities=forbidden
        )
        _reject_synthetic_opened_production(
            bundle_record, "bundle", require_production=self._require_production
        )
        if bundle_record["sha256"] != parent_binding["bundle_sha256"] or (
            bundle_record["bytes"] != parent_binding["bundle_bytes"]
        ):
            raise RuntimeError("validation loader bundle bytes differ from authority parent")
        payload = torch.load(
            io.BytesIO(bundle_raw), map_location="cpu", weights_only=True
        )
        fit_bundle.validate_fit_bundle_payload(
            payload, require_production=self._require_production
        )

        manifest_record, manifest_raw = parent._stable_record(
            self._paths.manifest, forbidden_identities=forbidden
        )
        _reject_synthetic_opened_production(
            manifest_record, "manifest", require_production=self._require_production
        )
        if manifest_record["sha256"] != parent_binding["manifest_artifact_sha256"]:
            raise RuntimeError("validation loader manifest differs from authority parent")
        manifest = parent._plain_json(manifest_raw, "factor validation FIT manifest")
        summary = fit_bundle.fit_bundle_manifest_summary_from_payload(
            payload,
            expected_authority_sha256=parent_binding["authority_logical_sha256"],
            require_production=self._require_production,
        )
        if manifest.get("bundle_summary") != summary:
            raise RuntimeError("validation loader manifest summary does not replay")

        receipt_record, receipt_raw = parent._stable_record(
            self._paths.receipt, forbidden_identities=forbidden
        )
        _reject_synthetic_opened_production(
            receipt_record, "receipt", require_production=self._require_production
        )
        if receipt_record["sha256"] != parent_binding["receipt_sha256"]:
            raise RuntimeError("validation loader receipt differs from authority parent")
        receipt = parent._plain_json(receipt_raw, "factor validation FIT receipt")
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
            raise RuntimeError("validation loader receipt does not join the bundle")
        checkpoint = receipt_payload.get("checkpoint")
        if type(checkpoint) is not dict or (
            checkpoint.get("config_sha256") != binding["config_sha256"]
            or checkpoint.get("weights_sha256") != binding["weights_sha256"]
        ):
            raise RuntimeError("validation loader checkpoint does not join the bundle")

        # Terminal replay after every semantic and deserialization check. The reducer
        # receives the private payload once; only cloned validation tensors return.
        if rebinding.fit_parent_binding_by_content_identity(self._paths) != parent_binding:
            raise RuntimeError("FIT parent changed during validation tensor load")
        result = validation_input_from_fit_payload(
            payload,
            parent_binding=parent_binding,
            candidate_freeze=candidate_freeze,
            candidate_freeze_audit=candidate_freeze_audit,
            candidate_freeze_artifact_sha256=candidate_freeze_artifact_sha256,
            candidate_freeze_audit_artifact_sha256=candidate_freeze_audit_artifact_sha256,
            require_production=self._require_production,
            train_documents=self._train_documents,
        )
        del payload
        del bundle_raw
        return result
