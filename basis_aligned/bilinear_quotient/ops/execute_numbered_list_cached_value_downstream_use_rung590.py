#!/usr/bin/env python3
"""Hash-pinned managed adapter for the prospective R590 replication."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"
PRODUCER = OPS / "numbered_list_cached_value_downstream_use_rung590.py"
OWNER_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung590.py"
DRYRUN = ROOT / "numbered_list_cached_value_downstream_use_rung590_dryrun.json"
NOTE = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG590_PROSPECTIVE_CONTRACT_REPLICATION.md"
BLOCK_REVIEW = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG584_POSTEXECUTION_CONTRACT_AUDIT.md"
BLOCK_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung584_postexecution_contract_audit.py"
HANDOFF_V1 = OPS / "circuit_causal_validity_next_wave_handoff_rung585.json"
HANDOFF_V2 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v2_addendum.json"
HANDOFF_V3 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v3_addendum.json"
HANDOFF_V4 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v4_addendum.json"
HANDOFF_V5 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"

RESULT = ROOT / "numbered_list_cached_value_downstream_use_rung590_results.json"
RECEIPT = ROOT / "numbered_list_cached_value_downstream_use_rung590_receipt.json"
EVIDENCE = ROOT / "numbered_list_cached_value_downstream_use_rung590_evidence"
OUTCOME_NAMESPACES = (RESULT, RECEIPT, EVIDENCE)

FROZEN_HASHES = {
    PRODUCER: "74b565fe835ee69a73ed1bdcdc103df3b2f4aa94931796ca1b96a4080639062e",
    OWNER_TEST: "037c7b7368fd2ca1f2d4656b75fd4e97e96e71c9c4f7679730ab94442fa6cee2",
    DRYRUN: "fb0b65d32be3422440602ae6458a39c357a1c83d5d180d0142f9f453edae3ad9",
    NOTE: "8b4019b2da24ee8a6acf73cf1cb35b157e3feece713ca9e90698a0801cf15ab5",
    BLOCK_REVIEW: "2fbefdb84822f4b727de769736f182f1b0864912c9f41f76247cc2df385cb45d",
    BLOCK_TEST: "8508b56c1c9e3d25ccd5f8b4cae0780fc263d0782682d8c57cdc22e8aaaef020",
    HANDOFF_V1: "e8970f9ef2d7eb7b291a5fb288833bc252e62fabf1016a699e981c19a6be560a",
    HANDOFF_V2: "eb8ef7d00324c7f38210f0e8303951d97282fc8dbede9ee10ef8409db414709b",
    HANDOFF_V3: "bf04cda987fc281f146c1e6f054620934f1d994a5d6d3135d7456be6fe9feb8c",
    HANDOFF_V4: "349afa9ec4fe465dbf08109a63cb1a8dc2a278e53a710bf210035f57b8500da0",
    HANDOFF_V5: "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80",
}

REGISTERED_PREDICATES = {
    "pred_a_frozen_bytes_match": "all prospective R590 and upstream contract bytes match",
    "pred_b_shape_contract_passes": "all 510 possible forward calls have compatible dynamic shapes",
    "pred_c_recovery_precedes_namespace_guard": "recognized stale packages reach quarantine first",
    "pred_d_science_command_is_explicit": "real execution invokes only --execute-science",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_bytes(
    bindings: Mapping[Path, str] = FROZEN_HASHES,
) -> dict[str, str]:
    observed = {}
    for path, expected in bindings.items():
        if not path.is_file():
            raise RuntimeError(f"frozen R590 file is missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"frozen R590 file changed: {path}; expected={expected}, observed={digest}"
            )
        observed[str(path)] = digest
    return observed


def load_frozen_producer():
    os.environ["BQLIB_NO_MODEL"] = "1"
    spec = importlib.util.spec_from_file_location("r590_managed_producer", PRODUCER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def require_unused_namespaces(paths: Sequence[Path] = OUTCOME_NAMESPACES) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RuntimeError(f"R590 output namespace already exists: {occupied}")


def run_model_free_validation() -> dict[str, object]:
    before = sha256(DRYRUN)
    producer = load_frozen_producer()
    plan = producer.run_dryrun()
    producer.validate_dryrun(plan)
    if plan != producer.strict_load_json(DRYRUN):
        raise RuntimeError("committed R590 dry run differs from isolated validation")
    if sha256(DRYRUN) != before:
        raise RuntimeError("committed R590 dry run changed during managed validation")
    shape = plan["forward_call_shape_contract"]
    if shape["call_count"] != 510 or shape["validation_mode"] != producer.SHAPE_MODE:
        raise RuntimeError("R590 forward-call shape contract is incomplete")
    support = plan["phase_support_census"]
    if plan["phase_support_census_sha256"] != producer.canonical_sha256(support):
        raise RuntimeError("R590 phase-specific support census hash changed")
    return plan


def scientific_command() -> tuple[str, list[str]]:
    return sys.executable, [sys.executable, str(PRODUCER), "--execute-science"]


def preflight(
    *, recovery_function: Callable[[], None] | None = None,
    namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
) -> dict[str, object]:
    observed = verify_frozen_bytes()
    producer = None
    if recovery_function is None:
        producer = load_frozen_producer()
        recovery_function = producer.recover_stale_publication
    recovery_function()
    require_unused_namespaces(namespace_paths)
    if producer is None:
        producer = load_frozen_producer()
    plan = producer.run_dryrun()
    return {
        "schema": "execute_numbered_list_cached_value_downstream_use_rung590_plan_v1",
        **{key: True for key in REGISTERED_PREDICATES},
        "frozen_sha256": observed,
        "forward_call_shape_contract": plan["forward_call_shape_contract"],
        "outcome_namespaces": [str(path) for path in namespace_paths],
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
    }


def dispatch(
    environment: Mapping[str, str], *,
    dry_validator: Callable[[], dict[str, object]] = run_model_free_validation,
    exec_function: Callable[[str, list[str]], object] = os.execv,
    recovery_function: Callable[[], None] | None = None,
    namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
) -> dict[str, object]:
    report = preflight(
        recovery_function=recovery_function, namespace_paths=namespace_paths
    )
    mode = environment.get("BQLIB_DRYRUN")
    if mode == "1":
        dry = dry_validator()
        report["mode"] = "model_free_dryrun"
        report["dryrun_status"] = dry["status"]
        report["next_step"] = "independent_preexecution_review_required"
        return report
    if mode is not None:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")
    executable, argv = scientific_command()
    exec_function(executable, argv)
    raise RuntimeError("R590 scientific os.execv unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("R590 managed adapter accepts no command-line arguments")
    report = dispatch(os.environ)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
