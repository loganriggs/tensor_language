#!/usr/bin/env python3
"""Hash-pinned managed adapter for the stdout-only R591 diagnostic."""

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

PRODUCER = OPS / "induction_replay_native_numerics_rung591.py"
PRODUCER_TEST = OPS / "test_induction_replay_native_numerics_rung591.py"
PRODUCER_DRYRUN = ROOT / "induction_replay_native_numerics_rung591_dryrun.json"
PREREGISTRATION = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_PREREGISTRATION.md"
BUILDER_HANDOFF = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_BUILDER_HANDOFF.md"
R585 = OPS / "induction_selector_payload_frozen_factor_rung585.py"
R585_TEST = OPS / "test_induction_selector_payload_frozen_factor_rung585.py"
R585_DRYRUN = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
FACADE = POLY / "bilin18_observed_model_facade.py"
INDUCTION = POLY / "circuit_induction_tensor.py"
MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
DEPENDENCY_LOCK = ROOT / "induction_selector_payload_frozen_factor_rung585_dependency_lock.json"
METHOD_HANDOFF_V5 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"

FROZEN_HASHES = {
    PRODUCER: "b2b266529f0f842211fea46856064133df5e3f4a8a7758c9095e7d29a94b6c49",
    PRODUCER_TEST: "e756ba3d17d3ebee2f81e97e573dd216090555de1fd3f1cfc926268f902d9ce7",
    PRODUCER_DRYRUN: "161193de5d90da69aafcd681e375993fa91d32e99100f0ed02fb586d5a629d8b",
    PREREGISTRATION: "e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593",
    BUILDER_HANDOFF: "61f8fb407dc026a7a2b126f2dce02b60266d040ffcce7159c5dc6a0d2517cc4f",
    R585: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    R585_TEST: "fcaba664269de12a41a5adb8ff089fc9963eeec91577ef94993ff032c02fc885",
    R585_DRYRUN: "580a570426ce48c9e43f5fce82c976dece6c71e8a11c1b057054c17cf958dcf8",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    INDUCTION: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    DEPENDENCY_LOCK: "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7",
    METHOD_HANDOFF_V5: "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80",
}

R585_NAMESPACES = (
    ROOT / "induction_selector_payload_frozen_factor_rung585_results.json",
    ROOT / "induction_selector_payload_frozen_factor_rung585_receipt.json",
    ROOT / "induction_selector_payload_frozen_factor_rung585_evidence",
)
R591_NAMESPACES = (
    ROOT / "induction_replay_native_numerics_rung591_results.json",
    ROOT / "induction_replay_native_numerics_rung591_receipt.json",
    ROOT / "induction_replay_native_numerics_rung591_evidence",
)
SCIENTIFIC_NAMESPACES = (*R585_NAMESPACES, *R591_NAMESPACES)

REGISTERED_PREDICATES = {
    "pred_a_exact_candidate_bytes": "all R591 candidate and method-dependency bytes match",
    "pred_b_namespaces_unoccupied": "R585 and R591 scientific namespaces are absent",
    "pred_c_model_free_dryrun": "managed preflight executes only R591 CPU validation",
    "pred_d_exact_diagnostic_exec": "the real branch execs only the pinned R591 diagnostic",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_bytes(
    bindings: Mapping[Path, str] = FROZEN_HASHES,
) -> dict[str, str]:
    observed = {}
    for path, expected in bindings.items():
        if not path.is_file():
            raise RuntimeError(f"frozen R591 file missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"frozen R591 file changed: {path}; expected {expected}, observed {digest}"
            )
        observed[str(path)] = digest
    return observed


def require_unused_namespaces(
    paths: Sequence[Path] = SCIENTIFIC_NAMESPACES,
) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RuntimeError(f"scientific namespace already exists: {occupied}")


def load_frozen_producer():
    os.environ["BQLIB_NO_MODEL"] = "1"
    name = "r591_hash_pinned_managed_producer"
    spec = importlib.util.spec_from_file_location(name, PRODUCER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen R591 producer: {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_model_free_validation() -> dict[str, object]:
    producer = load_frozen_producer()
    payload = producer.build_dryrun()
    committed = json.loads(PRODUCER_DRYRUN.read_text(encoding="utf-8"))
    if payload != committed:
        raise RuntimeError("R591 model-free dry run differs from committed receipt")
    if payload.get("model_forwards") != 0 or payload.get("model_backwards") != 0 or (
        payload.get("model_weights_updated") is not False
    ):
        raise RuntimeError("R591 dry run unexpectedly reports model work")
    if payload.get("scientific_status") != "diagnostic_only_no_scientific_terminal":
        raise RuntimeError("R591 dry run lost its non-scientific boundary")
    return payload


def diagnostic_command() -> tuple[str, list[str]]:
    return sys.executable, [sys.executable, str(PRODUCER)]


def preflight(
    *, namespace_paths: Sequence[Path] = SCIENTIFIC_NAMESPACES,
) -> dict[str, object]:
    observed = verify_frozen_bytes()
    require_unused_namespaces(namespace_paths)
    return {
        "schema": "execute_induction_replay_native_numerics_rung591_plan_v1",
        **{key: True for key in REGISTERED_PREDICATES},
        "frozen_sha256": observed,
        "scientific_namespaces": [str(path) for path in SCIENTIFIC_NAMESPACES],
        "registered_diagnostic_forwards": 234,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
    }


def dispatch(
    environment: Mapping[str, str], *,
    dry_validator: Callable[[], dict[str, object]] = run_model_free_validation,
    exec_function: Callable[[str, list[str]], object] = os.execv,
    namespace_paths: Sequence[Path] = SCIENTIFIC_NAMESPACES,
) -> dict[str, object]:
    plan = preflight(namespace_paths=namespace_paths)
    mode = environment.get("BQLIB_DRYRUN")
    if mode == "1":
        dryrun = dry_validator()
        plan["mode"] = "model_free_dryrun"
        plan["dryrun_schema"] = str(dryrun["schema"])
        plan["next_step"] = "different_agent_review_required"
        return plan
    if mode is not None:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")
    executable, argv = diagnostic_command()
    exec_function(executable, argv)
    raise RuntimeError("diagnostic os.execv unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("R591 managed adapter accepts no command-line arguments")
    report = dispatch(os.environ)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
