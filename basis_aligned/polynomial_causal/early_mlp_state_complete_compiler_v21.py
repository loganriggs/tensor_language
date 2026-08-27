#!/usr/bin/env python3
"""Fail-closed artifact lifecycle for the compiler-v2.1 numerical runner.

The CUDA capture/scoring code supplies complete candidate banks, controls, and
diagnostics to this module.  This module owns the irreversible ordering rules:
write and reload each nonauthorizing ledger before selection, recompute both
selectors, assemble the autoregressive program bundle, and write the final
unlock receipt last.  It never loads compiler_final_v21.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Mapping

import torch

import prepare_state_complete_compiler_rows_v21 as authority


ROOT = authority.ROOT
PROTOCOL = authority.PROTOCOL
AMENDMENT = authority.IMPLEMENTATION_AMENDMENT
ROWS_RECEIPT = authority.RECEIPT
PROGRAMS_ARTIFACT = authority.PROGRAMS_ARTIFACT
PROGRAMS_RECEIPT = authority.PROGRAMS_RECEIPT
SITE0_TRAINING_RECEIPT = authority.SITE0_TRAINING_RECEIPT
RUN_LOCK = Path("/workspace/runs/.early_mlp_state_complete_compiler_v21.lock")
STAGE_PATHS = {
    "site0": (authority.SITE0_LEDGER_ARTIFACT, authority.SITE0_LEDGER_RECEIPT),
    "site1": (authority.SITE1_LEDGER_ARTIFACT, authority.SITE1_LEDGER_RECEIPT),
}
STAGE_LEDGERS = {
    "site0": ("true_site0", "shuffle_site0"),
    "site1": ("true_site1", "shuffle_site1"),
}
DOWNSTREAM = {
    "site0": (
        SITE0_TRAINING_RECEIPT,
        authority.SITE1_LEDGER_ARTIFACT, authority.SITE1_LEDGER_RECEIPT,
        PROGRAMS_ARTIFACT, PROGRAMS_RECEIPT,
    ),
    "site1": (PROGRAMS_ARTIFACT, PROGRAMS_RECEIPT),
}


@dataclass(frozen=True)
class LaunchState:
    protected: tuple[tuple[str, str | None], ...]
    source_commit: str
    source_hashes: tuple[tuple[str, str], ...]
    rows_receipt_sha256: str
    rows_manifest_sha256: str
    lock_nonce: str = ""


@dataclass(frozen=True)
class ExecutionClosure:
    outer_model_returned: bool
    hook_restored_and_inert: bool
    component_tree_before: str
    component_tree_after: str


@contextmanager
def exclusive_run_claim():
    """Hold a create-only single-writer claim for the entire numerical pipeline."""

    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(24)
    payload = json.dumps({"pid": os.getpid(), "nonce": nonce}, sort_keys=True)
    try:
        descriptor = os.open(RUN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"compiler-v2.1 already claimed: {RUN_LOCK}") from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield nonce
    finally:
        try:
            current = json.loads(RUN_LOCK.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            current = None
        if current == {"pid": os.getpid(), "nonce": nonce}:
            RUN_LOCK.unlink()


def _require_run_claim(nonce: str) -> None:
    try:
        current = json.loads(RUN_LOCK.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("compiler-v2.1 exclusive run claim is absent") from error
    if not nonce or current != {"pid": os.getpid(), "nonce": nonce}:
        raise RuntimeError("compiler-v2.1 exclusive run claim changed")


def _source_identity() -> tuple[str, dict[str, str]]:
    """Return the synchronized committed program closure or fail before scoring."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != origin:
        raise RuntimeError("compiler-v2.1 requires HEAD==origin/main")
    hashes = {}
    for path in authority.PROGRAM_SOURCE_CLOSURE:
        authority.require_committed_clean(path)
        relative = path.resolve().relative_to(ROOT.resolve())
        hashes[str(relative)] = authority.file_sha256(path)
    return head, hashes


def verify_launch(*, lock_nonce: str) -> LaunchState:
    """Verify the committed launch boundary without deserializing validation/final."""

    _require_run_claim(lock_nonce)
    if authority.file_sha256(PROTOCOL) != authority.PINS[PROTOCOL] or (
        authority.file_sha256(AMENDMENT) != authority.IMPLEMENTATION_AMENDMENT_SHA256
    ) or authority.file_sha256(ROWS_RECEIPT) != authority.ROWS_RECEIPT_SHA256:
        raise RuntimeError("compiler-v2.1 launch authority changed")
    outputs = (
        *(path for pair in STAGE_PATHS.values() for path in pair),
        SITE0_TRAINING_RECEIPT, PROGRAMS_ARTIFACT, PROGRAMS_RECEIPT,
    )
    if any(path.exists() for path in outputs):
        raise RuntimeError("compiler-v2.1 output namespace is not empty")
    source_commit, source_hashes = _source_identity()
    protected = authority.protected_snapshot()
    authority._validate_historical_row_authority(json.loads(ROWS_RECEIPT.read_text()))
    return LaunchState(
        protected=tuple(sorted(protected.items())),
        source_commit=source_commit,
        source_hashes=tuple(sorted(source_hashes.items())),
        rows_receipt_sha256=authority.file_sha256(ROWS_RECEIPT),
        rows_manifest_sha256=authority.file_sha256(authority.MANIFEST),
        lock_nonce=lock_nonce,
    )


def resume_after_site0(*, lock_nonce: str) -> LaunchState:
    """Reconstruct launch authority for site1 from a completed site0 receipt."""

    _require_run_claim(lock_nonce)
    if authority.file_sha256(PROTOCOL) != authority.PINS[PROTOCOL] or (
        authority.file_sha256(AMENDMENT) != authority.IMPLEMENTATION_AMENDMENT_SHA256
    ) or authority.file_sha256(ROWS_RECEIPT) != authority.ROWS_RECEIPT_SHA256:
        raise RuntimeError("compiler-v2.1 resume authority changed")
    required = (*STAGE_PATHS["site0"], SITE0_TRAINING_RECEIPT)
    forbidden = (*STAGE_PATHS["site1"], PROGRAMS_ARTIFACT, PROGRAMS_RECEIPT)
    if any(not path.is_file() for path in required) or any(
        path.exists() for path in forbidden
    ):
        raise RuntimeError("compiler-v2.1 site0 resume output state changed")
    source_commit, source_hashes = _source_identity()
    protected = authority.protected_snapshot()
    authority._validate_historical_row_authority(json.loads(ROWS_RECEIPT.read_text()))
    receipt = load_site0_training_authorization()
    if receipt.get("source_commit") != source_commit or receipt.get(
        "source_hashes"
    ) != source_hashes:
        raise RuntimeError("compiler-v2.1 site0 source closure differs at resume")
    return LaunchState(
        protected=tuple(sorted(protected.items())),
        source_commit=source_commit,
        source_hashes=tuple(sorted(source_hashes.items())),
        rows_receipt_sha256=authority.file_sha256(ROWS_RECEIPT),
        rows_manifest_sha256=authority.file_sha256(authority.MANIFEST),
        lock_nonce=lock_nonce,
    )


def _validate_launch_state(state: LaunchState) -> None:
    if not isinstance(state, LaunchState):
        raise RuntimeError("v2.1 immutable launch state is absent")
    _require_run_claim(state.lock_nonce)
    source_commit, source_hashes = _source_identity()
    if source_commit != state.source_commit or tuple(sorted(source_hashes.items())) != (
        state.source_hashes
    ) or tuple(sorted(authority.protected_snapshot().items())) != state.protected or (
        authority.file_sha256(ROWS_RECEIPT) != state.rows_receipt_sha256
    ) or authority.file_sha256(authority.MANIFEST) != state.rows_manifest_sha256:
        raise RuntimeError("v2.1 launch-bound source/row/protected identity drifted")
    authority._validate_historical_row_authority(json.loads(ROWS_RECEIPT.read_text()))


def _require_inert_correction_state(sa: Any) -> None:
    from joint_early_mlp_oracle_factorial_authoritative import (  # noqa: PLC0415
        require_inert_correction_state,
    )

    require_inert_correction_state(sa)


def close_execution(
    sa: Any, *, outer_model_returned: bool, component_tree_before: str,
    component_tree_after: str,
) -> ExecutionClosure:
    """Measure inertness after outer return and mint an immutable closure token."""

    if not outer_model_returned or not component_tree_before or (
        component_tree_after != component_tree_before
    ):
        raise RuntimeError("v2.1 outer-return/component-tree lifecycle gate failed")
    _require_inert_correction_state(sa)
    return ExecutionClosure(
        outer_model_returned=True,
        hook_restored_and_inert=True,
        component_tree_before=component_tree_before,
        component_tree_after=component_tree_after,
    )


def _stage_paths(stage: str) -> tuple[Path, Path]:
    if stage not in STAGE_PATHS:
        raise ValueError(f"unknown compiler-v2.1 stage: {stage}")
    return STAGE_PATHS[stage]


def _validate_candidate_bank(name: str, bank: Any) -> None:
    specs = authority._candidate_specs()
    if not isinstance(bank, Mapping) or set(bank) != set(specs):
        raise RuntimeError(f"v2.1 {name} is not the exact 108-cell A-E bank")
    for candidate_name, record in bank.items():
        if not isinstance(record, Mapping) or set(record) != {"state", "metrics"}:
            raise RuntimeError(f"v2.1 malformed candidate record: {name}:{candidate_name}")
        authority._validate_candidate_state(
            candidate_name, record["state"], specs[candidate_name],
        )
        metrics = record["metrics"]
        if not isinstance(metrics, Mapping) or not {
            "recovery", "copy_worsening", "price",
        }.issubset(metrics):
            raise RuntimeError(f"v2.1 incomplete candidate metrics: {name}:{candidate_name}")
        numeric = torch.tensor([
            float(metrics["recovery"]), float(metrics["copy_worsening"]),
        ], dtype=torch.float64)
        if not bool(torch.isfinite(numeric).all()) or metrics[
            "price"
        ] != authority.selection.state_price(record["state"]):
            raise RuntimeError(f"v2.1 invalid candidate metrics: {name}:{candidate_name}")


def _validation_document_identity() -> str:
    receipt = json.loads(ROWS_RECEIPT.read_text())
    records = receipt.get("document_provenance", {}).get("sets", {}).get(
        "compiler_validation_v21"
    )
    if not isinstance(records, list) or not records or any(
        not isinstance(record, Mapping) or "document_id" not in record
        for record in records
    ):
        raise RuntimeError("v2.1 mapped-validation provenance is absent")
    return authority.logical_json_sha256([record["document_id"] for record in records])


def _validate_mean_state(value: Any, site: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "grammar", "interface", "family", "bias",
    } or value.get("grammar") != "constant" or value.get(
        "interface"
    ) != "state_complete_p" or value.get("family") != "fit_mean_control":
        raise RuntimeError(f"v2.1 mean-site{site} state changed")
    authority._finite_tensor(value.get("bias"), (authority.compiler.COEFFICIENT_DIM,))


def _selected_site0_programs() -> dict[str, Mapping[str, Any]]:
    payload, _ = load_frozen_stage("site0")
    selections = select_frozen_stage("site0")
    programs = {
        arm: payload["candidate_ledgers"][f"{arm}_site0"][
            selections[f"{arm}_site0"]["selected"]
        ]["state"]
        for arm in ("true", "shuffle")
    }
    programs["mean"] = payload["controls"]["mean_site0"]
    return programs


def _validate_stage_semantics(stage: str, payload: Mapping[str, Any]) -> None:
    """Apply amendment-level checks before a selector or downstream training."""

    ledgers = payload["candidate_ledgers"]
    controls = payload["controls"]
    diagnostics = payload["diagnostics"]
    expected_permutation = authority.expected_fit_permutation_sha256(
        json.loads(ROWS_RECEIPT.read_text())
    )
    capture_keys = ({
        "fit_original", "fit_shuffled", "validation_site0",
    } if stage == "site0" else {
        "true_fit_site1", "shuffle_fit_site1", "true_validation_site1",
        "shuffle_validation_site1", "mean_fit_site1",
    })
    captures = diagnostics["capture_hashes"]
    if diagnostics["fit_permutation_sha256"] != expected_permutation or not isinstance(
        captures, Mapping
    ) or set(captures) != capture_keys or any(
        not authority._is_sha256(value) for value in captures.values()
    ):
        raise RuntimeError(f"v2.1 {stage} capture/permutation binding changed")
    contexts = diagnostics["contexts"]
    if not isinstance(contexts, Mapping) or set(contexts) != set(STAGE_LEDGERS[stage]):
        raise RuntimeError(f"v2.1 {stage} context diagnostics changed")
    identity = _validation_document_identity()
    if stage == "site0":
        upstream = {"true_site0": "baseline", "shuffle_site0": "baseline"}
        _validate_mean_state(controls["mean_site0"], 0)
        authority._validate_full_native_control(
            controls["full_native_site0"], 0, context="baseline",
            upstream_state_sha256="baseline",
            validation_document_ids_sha256=identity,
        )
    else:
        receipt = load_site0_training_authorization()
        programs = _selected_site0_programs()
        upstream = {
            "true_site1": receipt["selected_state_sha256"]["true"],
            "shuffle_site1": receipt["selected_state_sha256"]["shuffle"],
        }
        _validate_mean_state(controls["mean_site1"], 1)
        authority._validate_full_native_control(
            controls["full_native_site1_true_context"], 1,
            context="true_site0", upstream_state_sha256=upstream["true_site1"],
            validation_document_ids_sha256=identity,
        )
        authority._validate_full_native_control(
            controls["full_native_site1_shuffle_context"], 1,
            context="shuffle_site0", upstream_state_sha256=upstream["shuffle_site1"],
            validation_document_ids_sha256=identity,
        )
        authority._validate_mean_site1_diagnostics(
            diagnostics["mean_control"], {
                "mean": {0: programs["mean"], 1: controls["mean_site1"]},
            },
        )
    for name, expected_upstream in upstream.items():
        authority._validate_context_diagnostics(
            contexts[name], stage=stage, name=name,
            expected_upstream=expected_upstream, candidates=ledgers[name],
        )
        authority._validate_candidate_sufficient_statistics(
            ledgers[name], contexts[name], name,
        )
    if stage == "site0":
        authority._validate_mean_score(
            diagnostics["mean_score"], contexts["true_site0"],
            stage="site0", expected_upstream="baseline",
        )


def _expected_stage_payload(
    stage: str, candidate_ledgers: Mapping[str, Any], controls: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    names = set(STAGE_LEDGERS[stage])
    if not isinstance(candidate_ledgers, Mapping) or set(candidate_ledgers) != names:
        raise RuntimeError(f"v2.1 {stage} candidate-ledger names changed")
    for name in names:
        _validate_candidate_bank(name, candidate_ledgers[name])
    expected_controls = ({"mean_site0", "full_native_site0"} if stage == "site0" else {
        "mean_site1", "full_native_site1_true_context",
        "full_native_site1_shuffle_context",
    })
    if not isinstance(controls, Mapping) or set(controls) != expected_controls:
        raise RuntimeError(f"v2.1 {stage} control schema changed")
    expected_diagnostics = {
        "fit_permutation_sha256", "capture_hashes", "contexts",
    } | ({"mean_control"} if stage == "site1" else {"mean_score"})
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != expected_diagnostics:
        raise RuntimeError(f"v2.1 {stage} diagnostic schema changed")
    payload = {
        "schema_version": 1,
        "status": f"pending_v21_{stage}_preselector_ledger",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "candidate_ledgers": dict(candidate_ledgers),
        "controls": dict(controls),
        "diagnostics": dict(diagnostics),
    }
    _validate_stage_semantics(stage, payload)
    return payload


def freeze_preselector_stage(
    stage: str, candidate_ledgers: Mapping[str, Any], controls: Mapping[str, Any],
    diagnostics: Mapping[str, Any], *, launch_state: LaunchState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write/reload a complete nonauthorizing stage ledger before selection."""

    artifact_path, receipt_path = _stage_paths(stage)
    if artifact_path.exists() or receipt_path.exists():
        raise RuntimeError(f"v2.1 {stage} stage output already exists")
    if any(path.exists() for path in DOWNSTREAM[stage]):
        raise RuntimeError(f"v2.1 {stage} downstream output exists before stage freeze")
    _validate_launch_state(launch_state)
    payload = _expected_stage_payload(stage, candidate_ledgers, controls, diagnostics)
    authority.write_torch_atomic(payload, artifact_path)
    _validate_launch_state(launch_state)
    reloaded = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not authority._same_value(reloaded, payload):
        raise RuntimeError(f"v2.1 {stage} external ledger did not reload exactly")
    _validate_launch_state(launch_state)
    receipt = {
        "status": f"frozen_v21_{stage}_preselector_ledger",
        "authority": f"compiler_v21_{stage}_preselector_ledger",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": authority.file_sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
    }
    authority.write_json_atomic(receipt, receipt_path)
    _validate_launch_state(launch_state)
    return load_frozen_stage(stage)


def load_frozen_stage(stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path, receipt_path = _stage_paths(stage)
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"v2.1 {stage} frozen stage is absent")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    receipt = json.loads(receipt_path.read_text())
    required = {
        "status": f"frozen_v21_{stage}_preselector_ledger",
        "authority": f"compiler_v21_{stage}_preselector_ledger",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": authority.file_sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
    }
    if not isinstance(receipt, Mapping) or set(receipt) != set(required) or any(
        receipt.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError(f"v2.1 {stage} receipt binding changed")
    expected = _expected_stage_payload(
        stage, payload.get("candidate_ledgers", {}), payload.get("controls", {}),
        payload.get("diagnostics", {}),
    ) if isinstance(payload, Mapping) else None
    if expected is None or not authority._same_value(payload, expected):
        raise RuntimeError(f"v2.1 {stage} artifact binding changed")
    return payload, dict(receipt)


def select_frozen_stage(stage: str) -> dict[str, dict[str, Any]]:
    """Recompute selectors only after the corresponding external ledger reloads."""

    payload, _ = load_frozen_stage(stage)
    true_name, shuffle_name = STAGE_LEDGERS[stage]
    return {
        true_name: authority.selection.freeze_validation_selection(
            payload["candidate_ledgers"][true_name]
        ),
        shuffle_name: authority._total_shuffle_selection(
            payload["candidate_ledgers"][shuffle_name]
        ),
    }


def _site0_training_receipt_candidate(
    launch_state: LaunchState, execution_closure: ExecutionClosure,
) -> dict[str, Any]:
    payload, _ = load_frozen_stage("site0")
    selections = select_frozen_stage("site0")
    programs = _selected_site0_programs()
    return {
        "status": "frozen_v21_site0_programs_after_outer_return",
        "authority": "compiler_v21_site0_to_site1_training_unlock",
        "authorized_for_training": True,
        "training_license_sites": [1],
        "authorized_for_final_scoring": False,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "stage_binding": _stage_binding("site0"),
        "selected": {
            arm: selections[f"{arm}_site0"]["selected"]
            for arm in ("true", "shuffle")
        },
        "selected_state_sha256": {
            arm: authority.state_logical_sha256(programs[arm])
            for arm in ("true", "shuffle")
        },
        "mean_state_sha256": authority.state_logical_sha256(
            payload["controls"]["mean_site0"]
        ),
        "component_tree_sha256": execution_closure.component_tree_after,
        "outer_model_returned": execution_closure.outer_model_returned,
        "hook_restored_and_inert": execution_closure.hook_restored_and_inert,
        "source_commit": launch_state.source_commit,
        "source_hashes": dict(launch_state.source_hashes),
    }


def write_site0_training_authorization_after_outer_return(
    *, launch_state: LaunchState, execution_closure: ExecutionClosure,
) -> dict[str, Any]:
    """Write the sole site1 training license after the site0 outer forward returns."""

    if not isinstance(execution_closure, ExecutionClosure) or not (
        execution_closure.outer_model_returned
        and execution_closure.hook_restored_and_inert
        and execution_closure.component_tree_before
        and execution_closure.component_tree_after == execution_closure.component_tree_before
    ):
        raise RuntimeError("v2.1 site0 training lifecycle gate failed")
    if SITE0_TRAINING_RECEIPT.exists() or any(
        path.exists() for path in (*STAGE_PATHS["site1"], PROGRAMS_ARTIFACT, PROGRAMS_RECEIPT)
    ):
        raise RuntimeError("v2.1 site0 training-unlock output state changed")
    _validate_launch_state(launch_state)
    candidate = _site0_training_receipt_candidate(launch_state, execution_closure)
    _validate_launch_state(launch_state)
    authority.write_json_atomic(candidate, SITE0_TRAINING_RECEIPT)
    return dict(candidate)


def load_site0_training_authorization() -> dict[str, Any]:
    if not SITE0_TRAINING_RECEIPT.is_file():
        raise RuntimeError("v2.1 site0-to-site1 training authority is absent")
    receipt = json.loads(SITE0_TRAINING_RECEIPT.read_text())
    programs = _selected_site0_programs()
    selections = select_frozen_stage("site0")
    required = {
        "status": "frozen_v21_site0_programs_after_outer_return",
        "authority": "compiler_v21_site0_to_site1_training_unlock",
        "authorized_for_training": True,
        "training_license_sites": [1],
        "authorized_for_final_scoring": False,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "stage_binding": _stage_binding("site0"),
        "selected": {
            arm: selections[f"{arm}_site0"]["selected"]
            for arm in ("true", "shuffle")
        },
        "selected_state_sha256": {
            arm: authority.state_logical_sha256(programs[arm])
            for arm in ("true", "shuffle")
        },
        "mean_state_sha256": authority.state_logical_sha256(programs["mean"]),
    }
    if not isinstance(receipt, Mapping) or not set(required).issubset(receipt) or any(
        not authority._same_value(receipt.get(key), value)
        for key, value in required.items()
    ) or set(receipt) != set(required) | {
        "component_tree_sha256", "outer_model_returned", "hook_restored_and_inert",
        "source_commit", "source_hashes",
    } or not receipt.get("component_tree_sha256") or receipt.get(
        "outer_model_returned"
    ) is not True or receipt.get("hook_restored_and_inert") is not True:
        raise RuntimeError("v2.1 site0-to-site1 training authority changed")
    return dict(receipt)


def _stage_binding(stage: str) -> dict[str, Any]:
    artifact, receipt = _stage_paths(stage)
    return {
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": authority.file_sha256(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "receipt_path": str(receipt.resolve()),
        "receipt_sha256": authority.file_sha256(receipt),
        "receipt_bytes": receipt.stat().st_size,
    }


def freeze_program_bundle(
    *, mean_programs: Mapping[int, Mapping[str, Any]], controls: Mapping[str, Any],
    strata: Mapping[str, Any], launch_state: LaunchState,
) -> dict[str, Any]:
    """Assemble, validate, and reload the nonauthorizing program artifact."""

    if PROGRAMS_ARTIFACT.exists() or PROGRAMS_RECEIPT.exists():
        raise RuntimeError("v2.1 program output already exists")
    _validate_launch_state(launch_state)
    site0_training_authorization = load_site0_training_authorization()
    site_payloads = {stage: load_frozen_stage(stage)[0] for stage in STAGE_PATHS}
    selections = {
        name: selection
        for stage in STAGE_PATHS
        for name, selection in select_frozen_stage(stage).items()
    }
    candidate_ledgers = {
        name: ledger
        for payload in site_payloads.values()
        for name, ledger in payload["candidate_ledgers"].items()
    }
    programs: dict[str, dict[int, Mapping[str, Any]]] = {
        "true": {}, "shuffle": {}, "mean": dict(mean_programs),
    }
    for arm in ("true", "shuffle"):
        for site in (0, 1):
            name = f"{arm}_site{site}"
            programs[arm][site] = candidate_ledgers[name][
                selections[name]["selected"]
            ]["state"]
    bundle = {
        "schema_version": 1,
        "status": "frozen_v21_program_bundle_pending_final_unlock",
        "authority": "compiler_v21_program_bundle",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "programs": programs,
        "pipeline_contexts": {
            "true": {0: "baseline", 1: "true_site0"},
            "shuffle": {0: "baseline", 1: "shuffle_site0"},
            "mean": {0: "baseline", 1: "mean_site0"},
        },
        "candidate_ledgers": candidate_ledgers,
        "selection_receipts": selections,
        "stage_bindings": {stage: _stage_binding(stage) for stage in STAGE_PATHS},
        "site0_training_authorization": {
            "path": str(SITE0_TRAINING_RECEIPT.resolve()),
            "sha256": authority.file_sha256(SITE0_TRAINING_RECEIPT),
            "bytes": SITE0_TRAINING_RECEIPT.stat().st_size,
            "receipt": site0_training_authorization,
        },
        "controls": dict(controls),
        "strata": dict(strata),
        "prices": {
            "true": authority._pipeline_price(programs["true"][0], programs["true"][1]),
            "shuffle": authority._pipeline_price(
                programs["shuffle"][0], programs["shuffle"][1]
            ),
            "mean": {
                "site0": authority._constant_price(),
                "site1": authority._constant_price(),
                "total_reals": 2 * authority._constant_price()["total_reals"],
            },
        },
    }
    authority._validate_program_bundle(bundle)
    _validate_launch_state(launch_state)
    authority.write_torch_atomic(bundle, PROGRAMS_ARTIFACT)
    _validate_launch_state(launch_state)
    reloaded = torch.load(PROGRAMS_ARTIFACT, map_location="cpu", weights_only=True)
    if not authority._same_value(reloaded, bundle):
        raise RuntimeError("v2.1 program bundle did not reload exactly")
    authority._validate_program_bundle(reloaded)
    _validate_launch_state(launch_state)
    return reloaded


def _validate_receipt_candidate(
    receipt: Mapping[str, Any], bundle: Mapping[str, Any], launch_state: LaunchState,
) -> None:
    _validate_launch_state(launch_state)
    authority._validate_program_bundle(bundle)
    required = {
        "status": "frozen_v21_programs_controls_strata_prices_before_final",
        "authority": "compiler_v21_final_unlock",
        "authorized_for_training": False,
        "authorized_for_final_scoring": True,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_path": str(ROWS_RECEIPT.resolve()),
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "programs_artifact_path": str(PROGRAMS_ARTIFACT.resolve()),
        "programs_artifact_sha256": authority.file_sha256(PROGRAMS_ARTIFACT),
        "programs_artifact_bytes": PROGRAMS_ARTIFACT.stat().st_size,
        "frozen_contents": {
            "true_program_sites": [0, 1],
            "shuffle_program_sites": [0, 1],
            "mean_program_sites": [0, 1],
            "candidate_ledgers_frozen": True,
            "controls_frozen": True,
            "strata_frozen": True,
            "standalone_prices_frozen": True,
            "preselector_stage_receipts_bound": True,
            "strata_derivations_recomputed": True,
            "site1_full_native_contexts": ["true", "shuffle"],
        },
        "source_commit": launch_state.source_commit,
        "source_hashes": dict(launch_state.source_hashes),
    }
    if set(receipt) != set(required) or any(
        not authority._same_value(receipt.get(key), value)
        for key, value in required.items()
    ):
        raise RuntimeError("v2.1 final receipt candidate changed")


def write_final_unlock_after_outer_return(
    *, launch_state: LaunchState, execution_closure: ExecutionClosure,
) -> dict[str, Any]:
    """Write the sole final authority after hook restoration and outer return."""

    if not isinstance(execution_closure, ExecutionClosure) or not (
        execution_closure.outer_model_returned
        and execution_closure.hook_restored_and_inert
        and execution_closure.component_tree_before
        and execution_closure.component_tree_after == execution_closure.component_tree_before
    ):
        raise RuntimeError("v2.1 outer-return/component-tree lifecycle gate failed")
    if PROGRAMS_RECEIPT.exists() or not PROGRAMS_ARTIFACT.is_file():
        raise RuntimeError("v2.1 final-unlock output state changed")
    _validate_launch_state(launch_state)
    bundle = torch.load(PROGRAMS_ARTIFACT, map_location="cpu", weights_only=True)
    authority._validate_program_bundle(bundle)
    receipt = {
        "status": "frozen_v21_programs_controls_strata_prices_before_final",
        "authority": "compiler_v21_final_unlock",
        "authorized_for_training": False,
        "authorized_for_final_scoring": True,
        "protocol_sha256": authority.PINS[PROTOCOL],
        "implementation_amendment_sha256": authority.IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_path": str(ROWS_RECEIPT.resolve()),
        "rows_receipt_sha256": authority.file_sha256(ROWS_RECEIPT),
        "programs_artifact_path": str(PROGRAMS_ARTIFACT.resolve()),
        "programs_artifact_sha256": authority.file_sha256(PROGRAMS_ARTIFACT),
        "programs_artifact_bytes": PROGRAMS_ARTIFACT.stat().st_size,
        "frozen_contents": {
            "true_program_sites": [0, 1],
            "shuffle_program_sites": [0, 1],
            "mean_program_sites": [0, 1],
            "candidate_ledgers_frozen": True,
            "controls_frozen": True,
            "strata_frozen": True,
            "standalone_prices_frozen": True,
            "preselector_stage_receipts_bound": True,
            "strata_derivations_recomputed": True,
            "site1_full_native_contexts": ["true", "shuffle"],
        },
        "source_commit": launch_state.source_commit,
        "source_hashes": dict(launch_state.source_hashes),
    }
    _validate_receipt_candidate(receipt, bundle, launch_state)
    authority.write_json_atomic(receipt, PROGRAMS_RECEIPT)
    return dict(receipt)


if __name__ == "__main__":
    raise SystemExit(
        "This lifecycle module is called by the audited CUDA numerical stages; "
        "it does not start a model forward by itself."
    )
