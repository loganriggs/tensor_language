#!/usr/bin/env python3
# BQLANE: cpu
"""Dry-run-first administrative closure of the abandoned Task14 DAS preregistration.

This publisher records no model outcome.  It only prevents the generated campaign
queue from presenting an explicitly stopped, never-executed implementation as live.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
import circuit_registry_v2 as registry  # noqa: E402


TAG = "task.subject_verb.number_agreement"
CLAIM_ID = "grammatical_subject_number.v10"
OPEN_EVENT = "task14_subject_number_localization.fit.preregistered.v2"
CLOSURE_EVENT = "task14_subject_number_localization.fit.v2.withdrawn.invalid.v1"
BASE_SHA256 = "66baec16c9da2ac27166c29fa12e011e1dcc4524c877ab26da01c07f089d4bed"

ARTIFACTS = {
    "localization_v2_compiler_v1_block_review": (
        "basis_aligned/polynomial_causal/"
        "TASK14_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_INDEPENDENT_REVIEW_2026-09-04.md",
        "673389c02ec4d7e9122557fe4fb44ab9f90950ccf8e6efbbd310ac6d543548b1",
        "independent_review",
    ),
    "localization_v2_compiler_v2_block_review": (
        "basis_aligned/polynomial_causal/"
        "TASK14_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_V2_INDEPENDENT_REVIEW_2026-09-04.md",
        "3131fffd0b6c8cd18789b69e4909b0002ca3e90f2c965391c07444f56b63756a",
        "independent_review",
    ),
    "localization_v2_administrative_stop": (
        "basis_aligned/polynomial_causal/explanations/explanation_2026-09-04_1330.md",
        "110b15d648626ecb07e26ebce89e6e85f7c322adfee927abc5e8bf9138e0bc03",
        "administrative_closure",
    ),
}
REQUIRED_EXISTING_ARTIFACTS = {
    "localization_v2_prereg": "3ea31387f611d0d095895dec6ed0859e1d99b2ad91a5d5adfb7be178bf127f59",
    "localization_v2_review": "2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384",
}


class PublicationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(relative: str, expected: str, kind: str) -> dict[str, str]:
    actual = _sha256(REPO / relative)
    if actual != expected:
        raise PublicationError(f"artifact hash mismatch for {relative}: {actual} != {expected}")
    return {"path": relative, "sha256": actual, "kind": kind, "status": "frozen"}


def _metric(name: str, estimate: object, bar: str) -> dict[str, object]:
    return {"name": name, "estimate": estimate, "ci95": None, "bar": bar}


def _base_from_current(current: dict, path: Path) -> dict:
    if _sha256(path) == BASE_SHA256:
        return current
    events = [event for event in current["evidence_events"] if event["event_id"] == CLOSURE_EVENT]
    if len(events) != 1:
        raise PublicationError("canonical Task14 record is not the exact v10 base or closure")
    base = copy.deepcopy(current)
    base["evidence_events"] = [
        event for event in base["evidence_events"] if event["event_id"] != CLOSURE_EVENT
    ]
    for artifact_id in ARTIFACTS:
        base["artifacts"].pop(artifact_id, None)
    claim = next(item for item in base["claims"] if item["claim_id"] == CLAIM_ID)
    claim["evidence_event_ids"] = [
        event_id for event_id in claim["evidence_event_ids"] if event_id != CLOSURE_EVENT
    ]
    payload = (json.dumps(base, indent=1) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != BASE_SHA256:
        raise PublicationError("applied Task14 closure does not reconstruct the exact v10 base")
    return base


def build_record(path: Path | None = None) -> dict:
    path = path or registry.circuit_path(TAG)
    current = json.loads(path.read_text())
    record = copy.deepcopy(_base_from_current(current, path))
    claim = next(item for item in record["claims"] if item["claim_id"] == CLAIM_ID)
    if claim is not record["claims"][-1]:
        raise PublicationError("Task14 v10 is no longer the latest canonical claim")
    next_missing = claim["next_missing"]
    prior_evidence_ids = list(claim["evidence_event_ids"])
    open_event = next(
        (event for event in record["evidence_events"] if event["event_id"] == OPEN_EVENT), None
    )
    if open_event is None or open_event["stage"] != "preregistered" \
            or open_event["verdict"] != "inconclusive":
        raise PublicationError("exact open Task14 localization-v2 preregistration is absent")
    if open_event["claim_id"] != "grammatical_subject_number.v2":
        raise PublicationError("open event is not the reviewed v2 design")

    for artifact_id, expected in REQUIRED_EXISTING_ARTIFACTS.items():
        artifact = record["artifacts"].get(artifact_id)
        if artifact is None or artifact.get("sha256") != expected:
            raise PublicationError(f"existing artifact binding changed: {artifact_id}")
        actual = _sha256(REPO / artifact["path"])
        if actual != expected:
            raise PublicationError(
                f"existing artifact hash mismatch for {artifact_id}: {actual} != {expected}"
            )

    for artifact_id, spec in ARTIFACTS.items():
        value = _artifact(*spec)
        if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
            raise PublicationError(f"artifact id collision: {artifact_id}")
        record["artifacts"][artifact_id] = value

    event = {
        "event_id": CLOSURE_EVENT,
        "claim_id": CLAIM_ID,
        "test_type": "das_interchange",
        "stage": "invalid",
        "verdict": "invalid",
        "failure_kind": "implementation_failure",
        "family_ids": list(open_event["family_ids"]),
        "site_id": None,
        "evaluation_role": (
            "administrative closure of the never-executed FIT localization-v2 plan after two "
            "independent compiler blocks and an explicit strategy stop"
        ),
        "metrics": [
            _metric("model_calls", 0, "must equal 0"),
            _metric("result_absent", True, "must be true"),
            _metric("compiler_v1_valid", False, "must be false"),
            _metric("compiler_v2_valid", False, "must be false"),
            _metric("compiler_v3_frozen", False, "must be false"),
        ],
        "result_artifact_id": "localization_v2_administrative_stop",
        "prereg_artifact_id": "localization_v2_prereg",
        "input_artifact_ids": [
            "localization_v2_review",
            "localization_v2_compiler_v1_block_review",
            "localization_v2_compiler_v2_block_review",
            "localization_v2_administrative_stop",
        ],
        "split_plan_id": open_event["split_plan_id"],
        "seed": open_event["seed"],
        "checkpoint_sha256": open_event["checkpoint_sha256"],
        "supersedes_event_id": OPEN_EVENT,
        "replicates_event_id": None,
        "sections": ["polynomial_causal/explanations/explanation_2026-09-04_1330.md"],
        "notes": (
            "Administrative implementation terminal only. The registered DAS intervention was "
            "never executed, so this event reports no learned coordinate, scientific null, task "
            "effect, CE effect, or model conclusion. Later Task14 fast-screen and below-head "
            "evidence are separate experiments and remain unchanged."
        ),
    }
    event["design_key"] = registry.design_key(record, event)
    event["execution_key"] = registry.execution_key(record, event)
    record["evidence_events"].append(event)
    claim["evidence_event_ids"] = [*prior_evidence_ids, CLOSURE_EVENT]
    if claim["next_missing"] != next_missing or claim["evidence_event_ids"][:-1] != prior_evidence_ids:
        raise PublicationError("v10 claim content changed beyond appending the closure evidence id")
    registry.validate_v2(record)
    return record


@contextmanager
def _target_lock(path: Path):
    lock_path = path.parent / f".{path.name}.administrative-closure.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def apply_record(
    record: dict | None = None, *, path: Path | None = None, regenerate: bool = True,
) -> Path:
    path = path or registry.circuit_path(TAG)
    expected = record or build_record(path)
    registry.validate_v2(expected)
    with _target_lock(path):
        current = json.loads(path.read_text())
        if current != expected:
            _base_from_current(current, path)  # exact-prefix guard
            registry._atomic_json(path, expected)
    if regenerate:
        if path.resolve() != registry.circuit_path(TAG).resolve():
            raise PublicationError("generated artifacts may only be rebuilt for the canonical record")
        registry.rebuild_registry_v2()
        for script in (
            "make_circuit_coverage.py",
            "make_circuit_experiment_index.py",
            "make_circuit_campaign_queue.py",
        ):
            subprocess.run([sys.executable, str(BQ / script)], cwd=REPO, check=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--record", type=Path, default=registry.circuit_path(TAG), help=argparse.SUPPRESS)
    parser.add_argument("--no-regenerate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    record = build_record(args.record)
    if args.apply:
        apply_record(record, path=args.record, regenerate=not args.no_regenerate)
    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "record": str(args.record),
        "event_id": CLOSURE_EVENT,
        "model_calls": 0,
        "scientific_outcome": None,
        "record_sha256": hashlib.sha256((json.dumps(record, indent=1) + "\n").encode()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
