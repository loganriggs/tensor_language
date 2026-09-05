#!/usr/bin/env python3
"""Publish the terminal R580/R581 and replacement R586/R587 capability chain."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
import circuit_registry_v2 as registry  # noqa: E402


TAG = "task.induction.selector_payload"
BASE_SHA256 = "c3d19b9dd3323948bd6546f17f6040767939827a50c903186ffdb7edc9a4567f"
OLD_CLAIM = "induction_selector_and_payload.v9"
NEW_CLAIM = "induction_selector_and_payload.v10"
OPEN_EVENT = "induction_selector_payload_native_capability.r580.preregistered.v2"
INVALID_EVENT = "induction_selector_payload_native_capability.r580.invalid_instrument.v1"
HELD_EVENT = "induction_selector_payload_native_capability.r586.complete.held.v1"
AUDIT_EVENT = "induction_selector_payload_native_capability_audit.r587.complete.held.v1"
NEXT_MISSING = (
    "adapt the already validated R557 selector-score/payload-value algebra and R558 interaction "
    "lattice to the frozen R578 rows in a separately preregistered factor experiment; do not "
    "repeat the native-capability screen"
)

ARTIFACTS = {
    "r580_capability_result": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung580_results.json",
        "7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84", "result"),
    "r580_capability_receipt": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung580_receipt.json",
        "6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a", "receipt"),
    "r581_capability_audit_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG581_PREREGISTRATION.md",
        "d2989383791cb179fecfa930742812cf8036a85bb9d2f3cfdd6555bb00640887", "preregistration"),
    "r581_capability_audit_result": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_audit_rung581.json",
        "8ecc1562632212ee876a794377e31966776ec15de02b5cb8d31798e438502cdb", "audit"),
    "r586_capability_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG586_PREREGISTRATION.md",
        "a139948085a99a6e745d3e8bf5d08ae11b58480d30ddf5e75467b506dda3a9a5", "preregistration"),
    "r586_capability_result": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json",
        "14e7414bc7cf6b4a6a221079ac378752602b021b8b411124149dcc2c311666b8", "result"),
    "r586_capability_receipt": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_receipt.json",
        "afd7533b1838b7d230858696a059f9c3a5903e75f031aa0c86f175f4bc0d9384", "receipt"),
    "r587_capability_audit_preregistration": (
        "basis_aligned/polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG587_PREREGISTRATION.md",
        "1f8e51ca7dcb4c8c9bb73ba13403c098871e13b593d995ff516ed839c2a9c771", "preregistration"),
    "r587_capability_audit_result": (
        "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_audit_rung587.json",
        "72f0261fe32aa3d048c442ea1c08af932af6a368894610833e79aaaabf98bfe9", "audit"),
}


class PublicationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_artifacts() -> dict[str, dict]:
    bound = {}
    for artifact_id, (relative, expected, kind) in ARTIFACTS.items():
        actual = _sha256(REPO / relative)
        if actual != expected:
            raise PublicationError(f"artifact hash mismatch for {artifact_id}: {actual} != {expected}")
        bound[artifact_id] = {
            "path": relative, "sha256": actual, "kind": kind, "status": "frozen",
        }
    return bound


def _metric(name: str, estimate: object, bar: str) -> dict:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _bind(record: dict, event: dict) -> dict:
    event["design_key"] = registry.design_key(record, event)
    event["execution_key"] = registry.execution_key(record, event)
    return event


def build_record(base: dict | None = None) -> dict:
    path = registry.circuit_path(TAG)
    if base is None:
        current = json.loads(path.read_text())
        if _sha256(path) == BASE_SHA256:
            base = current
        elif current.get("claims", [{}])[-1].get("claim_id") == NEW_CLAIM:
            # Reconstruct the only permitted v9 prefix so invoking the
            # publisher itself twice is byte-idempotent, not merely calling
            # apply_record(expected) twice in a unit test.
            candidate = copy.deepcopy(current)
            candidate["claims"] = candidate["claims"][:-1]
            candidate["evidence_events"] = candidate["evidence_events"][:-3]
            for artifact_id in ARTIFACTS:
                candidate["artifacts"].pop(artifact_id, None)
            payload = (json.dumps(candidate, indent=1) + "\n").encode()
            if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
                raise PublicationError("canonical induction record is not the exact v9/v10 chain")
            base = candidate
        else:
            raise PublicationError("canonical induction record is not the exact v9/v10 chain")
    record = copy.deepcopy(base)
    if record["claims"][-1]["claim_id"] != OLD_CLAIM:
        raise PublicationError("base record does not end at induction v9")
    if record["evidence_events"][-1]["event_id"] != OPEN_EVENT:
        raise PublicationError("base record does not end at the open R580 v2 event")
    for artifact_id, value in _bound_artifacts().items():
        if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
            raise PublicationError(f"artifact collision: {artifact_id}")
        record["artifacts"][artifact_id] = value

    previous = record["claims"][-1]
    claim = copy.deepcopy(previous)
    claim.update({
        "claim_id": NEW_CLAIM,
        "revision": 10,
        "supersedes": OLD_CLAIM,
        "status": "specified",
        "evidence_event_ids": [
            *previous["evidence_event_ids"], INVALID_EVENT, HELD_EVENT, AUDIT_EVENT,
        ],
        "next_missing": NEXT_MISSING,
    })
    record["claims"].append(claim)

    open_event = next(event for event in record["evidence_events"] if event["event_id"] == OPEN_EVENT)
    invalid = copy.deepcopy(open_event)
    invalid.update({
        "event_id": INVALID_EVENT,
        "claim_id": NEW_CLAIM,
        "stage": "invalid",
        "verdict": "invalid",
        "failure_kind": "invalid_instrument",
        "result_artifact_id": "r580_capability_result",
        "input_artifact_ids": list(dict.fromkeys([
            *open_event["input_artifact_ids"], "r580_capability_receipt",
            "r581_capability_audit_preregistration", "r581_capability_audit_result",
        ])),
        "supersedes_event_id": OPEN_EVENT,
        "replicates_event_id": None,
        "metrics": [
            _metric("scientific_capability_predicates", 3, "all 3 preregistered predicates hold"),
            _metric("independent_full_envelope_audit", 0, "must pass"),
        ],
        "notes": (
            "The native behavior screen was scientifically held, but R581 invalidated the result "
            "instrument because next_step was a one-item JSON list rather than the required scalar "
            "string. Preserve the scientific recomputation as descriptive evidence only."
        ),
    })

    held = copy.deepcopy(invalid)
    held.update({
        "event_id": HELD_EVENT,
        "stage": "complete",
        "verdict": "held",
        "failure_kind": None,
        "prereg_artifact_id": "r586_capability_preregistration",
        "result_artifact_id": "r586_capability_result",
        "input_artifact_ids": ["r586_capability_receipt"],
        # Operationally replace the invalid authority while explicitly retaining
        # that this is a prospective replication of its scientific design.
        "supersedes_event_id": INVALID_EVENT,
        "replicates_event_id": INVALID_EVENT,
        "seed": 586,
        "metrics": [
            _metric("scientific_capability_predicates", 3, "all 3 preregistered predicates hold"),
            _metric("unique_sequences", 3024, "must equal 3024"),
            _metric("execution_envelope", 95, "95 forwards, zero backwards/updates; FIT/SELECT only"),
            _metric("scalar_next_step", 1.0, "must be a scalar string"),
        ],
        "sections": [
            "polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_RUNG586_PREREGISTRATION.md"
        ],
        "notes": (
            "Prospective clean replication of R580 on the same frozen R578 rows and scientific "
            "contract. The result held with a corrected scalar next_step; FINAL_TEST/OOD stayed closed."
        ),
    })

    audit = {
        "event_id": AUDIT_EVENT, "claim_id": NEW_CLAIM, "test_type": "null_control",
        "stage": "complete", "verdict": "held", "failure_kind": None,
        "family_ids": held["family_ids"], "site_id": held["site_id"],
        "split_plan_id": held["split_plan_id"],
        "evaluation_role": "independent_model_free_raw_evidence_and_result_envelope_audit",
        "metrics": [
            _metric("raw_rows_recomputed", 3240, "must equal 3240"),
            _metric("factorial_groups_recomputed", 108, "must equal 108"),
            _metric("bootstrap_cells_recomputed", 86, "all 2,000-draw cells must match"),
            _metric("scientific_verdict_recomputed", 1.0, "must equal held_capability_screen"),
            _metric("result_receipt_binding", 1.0, "exact R586 result and receipt hashes must match"),
        ],
        "prereg_artifact_id": "r587_capability_audit_preregistration",
        "result_artifact_id": "r587_capability_audit_result",
        "input_artifact_ids": ["r586_capability_result", "r586_capability_receipt"],
        "seed": 587, "checkpoint_sha256": held["checkpoint_sha256"],
        "supersedes_event_id": None, "replicates_event_id": None,
        "sections": [
            "polynomial_causal/INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG587_PREREGISTRATION.md"
        ],
        "notes": (
            "Frozen before R586 outcomes. Independently reconstructed all raw evidence and scientific "
            "gates, and verified the strict envelope plus exact result/receipt byte binding with zero model calls."
        ),
    }
    record["evidence_events"].extend([
        _bind(record, invalid), _bind(record, held), _bind(record, audit),
    ])
    registry.validate_v2(record)
    return record


def apply_record(record: dict | None = None, *, regenerate: bool = True) -> Path:
    expected = build_record() if record is None else record
    registry.validate_v2(expected)
    path = registry.circuit_path(TAG)
    with registry._lock("registry"):
        existing = json.loads(path.read_text())
        if existing == expected:
            pass
        else:
            base = copy.deepcopy(expected)
            base["claims"] = base["claims"][:-1]
            base["evidence_events"] = base["evidence_events"][:-3]
            for artifact_id in ARTIFACTS:
                base["artifacts"].pop(artifact_id)
            if existing != base or _sha256(path) != BASE_SHA256:
                raise PublicationError("canonical record differs from exact v9 prefix")
            registry._atomic_json(path, expected)
    if regenerate:
        registry.rebuild_registry_v2()
    return path


def main() -> None:
    path = apply_record()
    print(json.dumps({"written": str(path.relative_to(REPO)), "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
