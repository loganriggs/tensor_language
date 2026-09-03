#!/usr/bin/env python3
"""Hash-pinned managed-execution adapter for the frozen R585 producer.

With ``BQLIB_DRYRUN=1`` this runs the producer's exact model-free validation
against a temporary dry-run output path.  With no mode variable it replaces
this process with the frozen producer's explicit scientific command.  Both
paths first require immutable inputs and unused R585 outcome namespaces.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"

REPAIR_BASE_COMMIT = "a4e7c46c6339c75fc7f89c1e35339e15e3b74fd9"
REPAIR_BASE_COMMIT_SHORT = "a4e7c46c6"
# Frozen by the independent BLOCK review.  These are provenance only; execution
# is pinned to the repaired hashes in FROZEN_HASHES below.
BLOCKED_PREDECESSOR_HASHES = {
    "producer": "4911200ae12dd9c27a609879fded8aab1b5704ef1116f25079b5df7a40162ff3",
    "owner_test": "71eab693b578478d39201c267cbea7311972602aec739de19de85acab59ca67e",
    "dryrun": "9b1b8c7c6e66a6b4835fa9ad10219fee16583f34d8a72c41a803cf6be5bfab7d",
}
PRODUCER = OPS / "induction_selector_payload_frozen_factor_rung585.py"
PRODUCER_TEST = OPS / "test_induction_selector_payload_frozen_factor_rung585.py"
PRODUCER_DRYRUN = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
DEPENDENCY_LOCK = ROOT / "induction_selector_payload_frozen_factor_rung585_dependency_lock.json"
MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
AMENDMENT = POLY / "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_REPLACEMENT_AMENDMENT.md"
BLOCKING_REVIEW = POLY / "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_PREREGISTRATION_REVIEW.md"
IMPLEMENTATION_REVIEW = POLY / (
    "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_IMPLEMENTATION_PREEXECUTION_REVIEW.md"
)
IMPLEMENTATION_ADVERSARIAL_TEST = OPS / (
    "test_induction_selector_payload_frozen_factor_rung585_implementation_adversarial.py"
)

RESULT = ROOT / "induction_selector_payload_frozen_factor_rung585_results.json"
RECEIPT = ROOT / "induction_selector_payload_frozen_factor_rung585_receipt.json"
EVIDENCE = ROOT / "induction_selector_payload_frozen_factor_rung585_evidence"

FROZEN_HASHES = {
    PRODUCER: "8a4f20d06dd04cd81d6bb8c94377ee987b66bea4201395e61bbe23a1b5dd9a8c",
    PRODUCER_TEST: "57e52e8da53f3a6e7b194efb64f56d1ff9fb442c2c39547a6f1fed4263a10653",
    PRODUCER_DRYRUN: "6fb41eb862c00f27673cfe694cf8670eae23f1d60a6a5dd85a35a5309b7e90f5",
    DEPENDENCY_LOCK: "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    AMENDMENT: "98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf",
    BLOCKING_REVIEW: "b8b4bcae6d2a24781383a5595a7c78d2d58623df209e9b98f7037ecc10566b2c",
    IMPLEMENTATION_REVIEW: "9bf8ae3c89d7c504bfdd42694771ef44bb87883429060d16335f0a1266d75a30",
    IMPLEMENTATION_ADVERSARIAL_TEST:
        "2567c3c5633575c2f4f8369328071025037b7c6f6c8a359f7870859b787a12e2",
}
OUTCOME_NAMESPACES = (RESULT, RECEIPT, EVIDENCE)

# These are adapter safety claims, not replacements for the producer's frozen
# scientific predicates.  Literal keys keep all four claims visible to gate.py.
REGISTERED_PREDICATES = {
    "pred_a_repair_base_and_bytes_match": "the R585 repair base and prospective execution bytes match",
    "pred_b_outcome_namespaces_are_unused": "all result, receipt, and evidence namespaces are absent",
    "pred_c_dryrun_is_model_free": "the dry-run branch executes only planted CPU validation",
    "pred_d_science_command_is_explicit": "the real branch invokes only the explicit science command",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_bytes(
    bindings: Mapping[Path, str] = FROZEN_HASHES,
) -> dict[str, str]:
    """Require every pinned file to exist and match its exact SHA-256."""
    observed = {}
    for path, expected in bindings.items():
        if not path.is_file():
            raise RuntimeError(f"frozen R585 file missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"frozen R585 file changed: {path}; expected {expected}, observed {digest}"
            )
        observed[str(path)] = digest
    return observed


def require_unused_namespaces(paths: Sequence[Path] = OUTCOME_NAMESPACES) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RuntimeError(f"R585 outcome namespace already exists: {occupied}")


def load_frozen_producer():
    """Import the already hash-checked producer without calling its CLI."""
    os.environ["BQLIB_NO_MODEL"] = "1"
    name = "r585_hash_pinned_managed_producer"
    spec = importlib.util.spec_from_file_location(name, PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen R585 producer: {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_model_free_validation() -> dict[str, object]:
    """Run the producer's complete dry-run checks without rewriting its receipt."""
    committed_dryrun_before = sha256(PRODUCER_DRYRUN)
    producer = load_frozen_producer()
    with tempfile.TemporaryDirectory(prefix="r585-managed-dryrun-") as tmp:
        isolated_receipt = Path(tmp) / PRODUCER_DRYRUN.name
        producer.DRYRUN = isolated_receipt
        payload = producer.run_dryrun()
        if not isolated_receipt.is_file():
            raise RuntimeError("frozen producer did not create its isolated dry-run receipt")
    if sha256(PRODUCER_DRYRUN) != committed_dryrun_before:
        raise RuntimeError("committed R585 dry-run receipt changed during validation")
    if payload.get("status") != "deterministic_cpu_dryrun_passed":
        raise RuntimeError("frozen producer's model-free validation did not pass")
    if payload.get("model_loaded") is not False or payload.get("cuda_opened") is not False:
        raise RuntimeError("R585 dry-run unexpectedly reported model or CUDA use")
    return payload


def scientific_command() -> tuple[str, list[str]]:
    argv = [sys.executable, str(PRODUCER), "--execute-science"]
    return sys.executable, argv


def recover_publication_preflight(recovery_function: Callable[[], None] | None = None) -> None:
    """Reach the hash-pinned producer's conservative stale-package recovery."""
    if recovery_function is None:
        recovery_function = load_frozen_producer().recover_stale_publication
    recovery_function()


def preflight(
    *, recovery_function: Callable[[], None] | None = None,
    namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
) -> dict[str, object]:
    observed = verify_frozen_bytes()
    recover_publication_preflight(recovery_function)
    require_unused_namespaces(namespace_paths)
    return {
        "schema": "execute_induction_selector_payload_frozen_factor_rung585_plan_v1",
        "repair_base_commit": REPAIR_BASE_COMMIT,
        "repair_base_commit_short": REPAIR_BASE_COMMIT_SHORT,
        **{key: True for key in REGISTERED_PREDICATES},
        "frozen_sha256": observed,
        "outcome_namespaces": [str(path) for path in OUTCOME_NAMESPACES],
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
    }


def dispatch(
    environment: Mapping[str, str],
    *,
    dry_validator: Callable[[], dict[str, object]] = run_model_free_validation,
    exec_function: Callable[[str, list[str]], object] = os.execv,
    recovery_function: Callable[[], None] | None = None,
    namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
) -> dict[str, object]:
    execution_plan = preflight(
        recovery_function=recovery_function, namespace_paths=namespace_paths
    )
    mode = environment.get("BQLIB_DRYRUN")
    if mode == "1":
        dryrun = dry_validator()
        execution_plan["mode"] = "model_free_dryrun"
        execution_plan["dryrun_status"] = dryrun["status"]
        execution_plan["next_step"] = "managed_preflight_complete"
        return execution_plan
    if mode is not None:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")

    executable, argv = scientific_command()
    exec_function(executable, argv)
    raise RuntimeError("scientific os.execv unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("R585 managed adapter accepts no command-line arguments")
    report = dispatch(os.environ)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
