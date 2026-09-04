#!/usr/bin/env python3
"""Hash-pinned managed adapter for the prospective R590 replication."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
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
HANDOFF_V6 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v6_addendum.json"
ADAPTER_TEST = OPS / "test_execute_numbered_list_cached_value_downstream_use_rung590.py"
R590_BLOCK_REVIEW = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG590_PREEXECUTION_REVIEW.md"
R590_BLOCK_TEST = OPS / "test_numbered_list_cached_value_downstream_use_rung590_preexecution_review_adversarial.py"
R584_RUNNER = OPS / "numbered_list_cached_value_downstream_use_rung584.py"
R588_AUDITOR = OPS / "audit_numbered_list_cached_value_downstream_use_rung588.py"
RESULT_CONTRACT = OPS / "result_contract.py"
FACADE = POLY / "bilin18_observed_model_facade.py"
R576_RUNNER = OPS / "numbered_list_cached_value_weight_removal_rung576.py"
R573_RUNNER = OPS / "numbered_list_factor_localization_rung573.py"
R582_HELPER = OPS / "numbered_list_cached_value_downstream_use_rung582.py"
JACCLUST_PACKAGE = ROOT.parent.parent / "jacclust" / "__init__.py"
TT_MODEL = ROOT.parent.parent / "jacclust" / "tt_model.py"

RESULT = ROOT / "numbered_list_cached_value_downstream_use_rung590_results.json"
RECEIPT = ROOT / "numbered_list_cached_value_downstream_use_rung590_receipt.json"
EVIDENCE = ROOT / "numbered_list_cached_value_downstream_use_rung590_evidence"
OUTCOME_NAMESPACES = (RESULT, RECEIPT, EVIDENCE)

FROZEN_HASHES = {
    PRODUCER: "5cc4544158312d7fa6224bf46c635acbb0d4a11fc2d620cedc2516d169f5966e",
    OWNER_TEST: "49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0",
    DRYRUN: "817f457ba1cc9737735182f495c54a3956be8c5dd6267bb5d8222f40e750d603",
    NOTE: "a6641a20a456d30895a9ba807c22ec74e7695fe5c84ce4300b909787c603afa7",
    BLOCK_REVIEW: "2fbefdb84822f4b727de769736f182f1b0864912c9f41f76247cc2df385cb45d",
    BLOCK_TEST: "8508b56c1c9e3d25ccd5f8b4cae0780fc263d0782682d8c57cdc22e8aaaef020",
    HANDOFF_V1: "e8970f9ef2d7eb7b291a5fb288833bc252e62fabf1016a699e981c19a6be560a",
    HANDOFF_V2: "eb8ef7d00324c7f38210f0e8303951d97282fc8dbede9ee10ef8409db414709b",
    HANDOFF_V3: "bf04cda987fc281f146c1e6f054620934f1d994a5d6d3135d7456be6fe9feb8c",
    HANDOFF_V4: "349afa9ec4fe465dbf08109a63cb1a8dc2a278e53a710bf210035f57b8500da0",
    HANDOFF_V5: "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80",
    HANDOFF_V6: "d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c",
    ADAPTER_TEST: "4c5bd25cdf06e21f823c9e09fdd57a7ca54d8700aa23a379a7913e2fc8c6b174",
    R590_BLOCK_REVIEW: "c3d4825695f0c3ca4b6fecf4d31dde67eb41b732286f783711d69c1d7c4ba9b7",
    R590_BLOCK_TEST: "23872cfac16a1adc47c3ff492fbac4ea63de5d1d5a030de3f282b91ce2e589a5",
    R584_RUNNER: "50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7",
    R588_AUDITOR: "b4acebb23bff71c7dc11beec95ff83f5490a86971787bce5930351cfb4572115",
    RESULT_CONTRACT: "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    R576_RUNNER: "91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a",
    R573_RUNNER: "5723e42e2a5f72a4ddab7a20b631e18e0b6d28875ff53f3db2d37d1845d6e076",
    R582_HELPER: "b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c",
    JACCLUST_PACKAGE: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    TT_MODEL: "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
}

EXECUTABLE_LOAD_ORDER = (
    ("jacclust", JACCLUST_PACKAGE, True),
    ("jacclust.tt_model", TT_MODEL, False),
    ("result_contract", RESULT_CONTRACT, False),
    ("bilin18_observed_model_facade", FACADE, False),
    ("numbered_list_factor_localization_rung573", R573_RUNNER, False),
    ("numbered_list_cached_value_weight_removal_rung576", R576_RUNNER, False),
    ("numbered_list_cached_value_downstream_use_rung582", R582_HELPER, False),
    ("numbered_list_cached_value_downstream_use_rung584", R584_RUNNER, False),
    ("audit_numbered_list_cached_value_downstream_use_rung588", R588_AUDITOR, False),
)

REGISTERED_PREDICATES = {
    "pred_a_frozen_bytes_match": "all prospective R590 and upstream contract bytes match",
    "pred_b_shape_contract_passes": "all 510 possible forward calls have compatible dynamic shapes",
    "pred_c_recovery_precedes_namespace_guard": "recognized stale packages reach quarantine first",
    "pred_d_science_entry_is_immutable": (
        "real execution calls --execute-science on the verified in-memory producer"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_frozen_bytes(
    bindings: Mapping[Path, str] = FROZEN_HASHES,
) -> dict[Path, bytes]:
    """Read every bound file once before importing any project code."""
    captured = {}
    for path, expected in bindings.items():
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as capture_exc:
            raise RuntimeError(f"frozen R590 file is missing or unsafe: {path}") from capture_exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"frozen R590 source is not a regular file: {path}")
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"frozen R590 file changed while being captured: {path}")
        data = b"".join(chunks)
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"frozen R590 file changed: {path}; expected={expected}, observed={digest}"
            )
        captured[path] = data
    return captured


def verify_frozen_bytes(
    bindings: Mapping[Path, str] = FROZEN_HASHES,
) -> dict[str, str]:
    snapshot = capture_frozen_bytes(bindings)
    return {str(path): hashlib.sha256(data).hexdigest() for path, data in snapshot.items()}


def _module_from_verified_bytes(
    name: str, path: Path, source: bytes, *, is_package: bool = False,
) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name if is_package else name.rpartition(".")[0]
    if is_package:
        module.__path__ = [str(path.parent)]
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child, module)
    return module


def load_frozen_producer(snapshot: Mapping[Path, bytes] | None = None):
    os.environ["BQLIB_NO_MODEL"] = "1"
    frozen = dict(capture_frozen_bytes() if snapshot is None else snapshot)
    missing = set(FROZEN_HASHES) - set(frozen)
    if missing:
        raise RuntimeError(f"verified snapshot is incomplete: {sorted(map(str, missing))}")
    for name, path, is_package in EXECUTABLE_LOAD_ORDER:
        _module_from_verified_bytes(name, path, frozen[path], is_package=is_package)
    return _module_from_verified_bytes(
        "r590_managed_producer", PRODUCER, frozen[PRODUCER]
    )


def require_unused_namespaces(paths: Sequence[Path] = OUTCOME_NAMESPACES) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RuntimeError(f"R590 output namespace already exists: {occupied}")


def run_model_free_validation(
    *, snapshot: Mapping[Path, bytes] | None = None, producer=None,
) -> dict[str, object]:
    frozen = dict(capture_frozen_bytes() if snapshot is None else snapshot)
    before = sha256(DRYRUN)
    if producer is None:
        producer = load_frozen_producer(frozen)
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


def run_verified_science(producer) -> None:
    producer.main(["--execute-science"])


def preflight(
    *, recovery_function: Callable[[], None] | None = None,
    namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
    snapshot: Mapping[Path, bytes] | None = None,
    producer=None,
) -> dict[str, object]:
    frozen = dict(capture_frozen_bytes() if snapshot is None else snapshot)
    observed = {
        str(path): hashlib.sha256(data).hexdigest() for path, data in frozen.items()
    }
    if producer is None:
        producer = load_frozen_producer(frozen)
    if recovery_function is None:
        recovery_function = producer.recover_stale_publication
    recovery_function()
    require_unused_namespaces(namespace_paths)
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
    dry_validator: Callable[[], dict[str, object]] | None = None,
    science_function: Callable[[object], object] = run_verified_science,
    recovery_function: Callable[[], None] | None = None,
    namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
) -> dict[str, object]:
    snapshot = capture_frozen_bytes()
    producer = load_frozen_producer(snapshot)
    report = preflight(
        recovery_function=recovery_function, namespace_paths=namespace_paths,
        snapshot=snapshot, producer=producer,
    )
    mode = environment.get("BQLIB_DRYRUN")
    if mode == "1":
        dry = (
            run_model_free_validation(snapshot=snapshot, producer=producer)
            if dry_validator is None else dry_validator()
        )
        report["mode"] = "model_free_dryrun"
        report["dryrun_status"] = dry["status"]
        report["next_step"] = "independent_preexecution_review_required"
        return report
    if mode is not None:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")
    science_function(producer)
    raise RuntimeError("R590 verified scientific entry point unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("R590 managed adapter accepts no command-line arguments")
    report = dispatch(os.environ)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
