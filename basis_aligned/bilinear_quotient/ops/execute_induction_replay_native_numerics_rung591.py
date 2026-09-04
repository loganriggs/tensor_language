#!/usr/bin/env python3
"""Hash-pinned managed adapter for the stdout-only R591 diagnostic."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import base64
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
R578_ROWS = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
R585_AMENDMENT = POLY / "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_REPLACEMENT_AMENDMENT.md"
R459_FACTOR = OPS / "equality_term_score_payload_rung459.py"
CANONICAL_TERM = OPS / "equality_term_subset_factorial_stage1.py"
METHOD_HANDOFF_V5 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"
METHOD_HANDOFF_V6 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v6_addendum.json"
TT_MODEL = ROOT.parent.parent / "jacclust" / "tt_model.py"

FROZEN_HASHES = {
    PRODUCER: "fb8239ded4f3e99510f37ea72337c2d69e4640f7a2556748c9062aa82b2751bc",
    PRODUCER_TEST: "8a24a9903d10ada8a4048c7adcb33cb4ef3e8aeef11d6f9718f8e50e57b6212c",
    PRODUCER_DRYRUN: "8a6331fb1a4d3800abff5ab6b7e291105872b06b41a43b003436312b6e50dc5d",
    PREREGISTRATION: "2dd8f918f767a6e5d91af357cfaa14770b79334ebac837d1bf52e8046ce190a5",
    BUILDER_HANDOFF: "202f1268e583a82f6cca385f4223b6edf4e8f8bbaee2c1cc975b09e51cd95f12",
    R585: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    R585_TEST: "fcaba664269de12a41a5adb8ff089fc9963eeec91577ef94993ff032c02fc885",
    R585_DRYRUN: "580a570426ce48c9e43f5fce82c976dece6c71e8a11c1b057054c17cf958dcf8",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    INDUCTION: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    DEPENDENCY_LOCK: "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7",
    R578_ROWS: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
    R585_AMENDMENT: "98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf",
    R459_FACTOR: "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
    CANONICAL_TERM: "3caa753cd856ec87899936fe71137ce28e893f86433558f40a815afff61824af",
    METHOD_HANDOFF_V5: "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80",
    METHOD_HANDOFF_V6: "d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c",
    TT_MODEL: "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
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
    source = PRODUCER.read_bytes()
    expected = FROZEN_HASHES[PRODUCER]
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError("R591 producer changed before immutable import")
    name = "r591_hash_pinned_managed_producer"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(PRODUCER))
    if spec is None:
        raise RuntimeError(f"cannot construct frozen R591 producer: {PRODUCER}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(PRODUCER)
    sys.modules[name] = module
    exec(compile(source, str(PRODUCER), "exec"), module.__dict__)
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


def diagnostic_command(
    producer: Path = PRODUCER,
    expected_sha256: str = FROZEN_HASHES[PRODUCER],
) -> tuple[str, list[str]]:
    """Embed the verified producer bytes so exec cannot reopen a swapped path."""
    source = producer.read_bytes()
    observed = hashlib.sha256(source).hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(
            f"producer changed before immutable dispatch: expected {expected_sha256}, "
            f"observed {observed}"
        )
    encoded = base64.b64encode(source).decode("ascii")
    logical_path = str(producer)
    launcher = (
        "import base64,sys;"
        f"_p={logical_path!r};sys.argv=[_p];"
        f"_b=base64.b64decode({encoded!r});"
        "exec(compile(_b,_p,'exec'),"
        "{'__name__':'__main__','__file__':_p,'__package__':None})"
    )
    return sys.executable, [sys.executable, "-I", "-c", launcher]


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
