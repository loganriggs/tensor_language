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
AUDIT = HERE / "mlp2_error_rayleigh_v1_design_scorer_independent_audit.json"
AUTHORITY = HERE / "mlp2_error_rayleigh_v1_design_predictor_authority.json"
BUNDLE = HERE / "mlp2_error_rayleigh_v1_design_predictor_bundle.pt"
RECEIPT = collector.PREDICTOR_RECEIPT
FAILURE = HERE / "mlp2_error_rayleigh_v1_design_predictor_failure.json"
LOCK = Path("/workspace/runs/.mlp2_error_rayleigh_v1_design_predictor.lock")
DESIGN = collector.role_paths("DESIGN")
SOURCE_PATHS = tuple(dict.fromkeys((
    collector.PREREG, collector.ADDENDUM,
    HERE / "mlp2_error_rayleigh_predictor.py",
    HERE / "test_mlp2_error_rayleigh_predictor.py",
    RUNNER, TEST,
    collector.CORE, collector.CORE_TEST, collector.RUNNER, collector.TEST,
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
        "mlp2_error_rayleigh_v1_design_scorer_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) \
            or value["tests_passed"] < 1 or not value.get("reviewer"):
        raise RuntimeError("DESIGN scorer audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("DESIGN scorer audit binding changed")
    return value, digest


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


def protected_snapshot(authority: Mapping[str, Any]) -> dict[str, str]:
    if source_hashes(authority["source_commit"]) != authority["source_hashes"]:
        raise RuntimeError("DESIGN scorer sources changed")
    _, audit_sha = validate_audit(authority["source_hashes"])
    receipt, receipt_sha = base.stable_json(DESIGN["receipt"], authority["design_receipt_sha256"])
    ledger, ledger_sha = base.stable_torch(DESIGN["ledger"], authority["design_ledger_sha256"])
    validate_design_receipt(receipt, ledger_sha)
    collector.validate_ledger(ledger, receipt["authority_sha256"], "DESIGN")
    if audit_sha != authority["audit_sha256"] or receipt_sha != authority["design_receipt_sha256"]:
        raise RuntimeError("DESIGN scorer protected hashes changed")
    return {"audit": audit_sha, "receipt": receipt_sha, "ledger": ledger_sha}


def serialize_fit(value: Mapping[str, Any]) -> dict[str, Any]:
    models = {}
    for name, model in value["models"].items():
        models[name] = {
            "ridge_selected": model["ridge"]["selected"],
            "clustered_lodo_mse": model["ridge"]["clustered_lodo_mse"],
            "mean": model["mean"].clone(), "scale": model["scale"].clone(),
            "coefficients": model["coefficients"].clone(),
            "design_prediction": model["design_prediction"].clone(),
        }
    return {
        "schema": "mlp2_error_rayleigh_v1_design_predictor_bundle",
        "target": value["target"].clone(), "models": models,
        "null_predictions": {
            control: {family: prediction.clone() for family, prediction in families.items()}
            for control, families in value["null_predictions"].items()
        },
        "families": {name: list(features) for name, features in predictor.FAMILIES.items()},
        "ridge_grid": list(predictor.RIDGE_GRID),
        "unit": "source_document_by_program",
        "program_identity_feature": False,
        "directional_amplitude_reduction": "arithmetic_mean_h16_h8",
    }


def validate_bundle(value: Any) -> dict[str, Any]:
    predictor.validate_frozen_bundle(value)
    return value


def run() -> None:
    paths = (AUTHORITY, BUNDLE, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in paths):
        raise RuntimeError("DESIGN predictor namespace already exists")
    if not DESIGN["receipt"].is_file() or not DESIGN["ledger"].is_file():
        raise RuntimeError("DESIGN measurements are not receipt-complete")
    claim = row_life.base.acquire_claim(LOCK)
    authority = None; opened = False
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        sources = source_hashes(commit); audit, audit_sha = validate_audit(sources)
        design_receipt_sha = file_sha256(DESIGN["receipt"])
        design_ledger_sha = file_sha256(DESIGN["ledger"])
        authority = {
            "schema": "mlp2_error_rayleigh_v1_design_predictor_authority",
            "status": "frozen_before_design_ledger_open", "source_commit": commit,
            "source_hashes": sources, "audit_sha256": audit_sha,
            "audit_reviewer": audit["reviewer"],
            "design_receipt_sha256": design_receipt_sha,
            "design_ledger_sha256": design_ledger_sha,
            "ridge_grid": list(predictor.RIDGE_GRID),
            "families": {name: list(features) for name, features in predictor.FAMILIES.items()},
            "heldout_opened": False,
        }
        protected = protected_snapshot(authority)

        def authority_guard():
            row_life.base.require_claim(claim, LOCK)
            if any(path.exists() for path in (AUTHORITY, BUNDLE, RECEIPT, FAILURE)) \
                    or protected_snapshot(authority) != protected:
                raise RuntimeError("DESIGN predictor authority inputs changed")
            row_life.base.require_claim(claim, LOCK)

        base.atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(AUTHORITY); opened = True
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
            if protected_snapshot(authority) != protected or BUNDLE.exists() \
                    or RECEIPT.exists() or FAILURE.exists():
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
            "predictor_bundle_sha256": bundle_sha, "heldout_unlocked": True,
        }

        def receipt_guard():
            if protected_snapshot(authority) != protected:
                raise RuntimeError("DESIGN predictor protected state changed")
            base.stable_json(AUTHORITY, authority_sha); base.stable_torch(BUNDLE, bundle_sha)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("DESIGN predictor terminal raced receipt")
            row_life.base.require_claim(claim, LOCK)

        base.atomic_json(RECEIPT, receipt, pre_link_check=receipt_guard)
        print(json.dumps(receipt, sort_keys=True, indent=2))
    except BaseException as exc:
        failure = {
            "schema": "mlp2_error_rayleigh_v1_design_predictor_failure",
            "status": "terminal_failure_no_receipt", "error": repr(exc),
            "authority_exists": AUTHORITY.exists(), "design_ledger_may_have_opened": opened,
        }
        if not RECEIPT.exists() and not FAILURE.exists():
            base.atomic_json(FAILURE, failure)
        raise
    finally:
        row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    run()
