#!/usr/bin/env python3
"""Immutable-byte managed preflight and scientific dispatch adapter for R593."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
POLY = ROOT.parent / "polynomial_causal"
ADAPTER = OPS / "execute_induction_centered_fixed_geometry_rung593.py"
PRODUCER = OPS / "induction_centered_fixed_geometry_rung593.py"
OWNER_TEST = OPS / "test_induction_centered_fixed_geometry_rung593.py"
FAKE_RUNTIME_TEST = OPS / "test_induction_centered_fixed_geometry_rung593_fake_runtime.py"
ADAPTER_TEST = OPS / "test_execute_induction_centered_fixed_geometry_rung593.py"
RUNTIME = OPS / "induction_centered_fixed_geometry_rung593_runtime.py"
DRYRUN = ROOT / "induction_centered_fixed_geometry_rung593_dryrun.json"
PREREG = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION.md"
AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_AMENDMENT.md"
DIAGNOSTIC_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT.md"
MASK_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT.md"
TOPOLOGY_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_LOGIT_TOPOLOGY_AMENDMENT.md"
TOPOLOGY_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_LOGIT_TOPOLOGY_AMENDMENT_INDEPENDENT_REVIEW.md"
TOPOLOGY_REVIEW_TEST = OPS / "test_induction_centered_fixed_geometry_rung592_logit_topology_amendment_review.py"
IMPLEMENTATION_BLOCK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_IMPLEMENTATION_PREEXECUTION_REVIEW.md"
IMPLEMENTATION_BLOCK_TEST = OPS / "test_induction_centered_fixed_geometry_rung592_implementation_preexecution_review.py"
STORAGE_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_STREAMING_STORAGE_AMENDMENT.md"
STORAGE_BLOCK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_REPAIR_PREEXECUTION_REVIEW.md"
STORAGE_BLOCK_TEST = OPS / "test_induction_centered_fixed_geometry_rung592_repair_preexecution_review.py"
CAPACITY_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PHASE_RELATIVE_CAPACITY_AMENDMENT.md"
STREAMING_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_STREAMING_PREEXECUTION_REVIEW.md"
STREAMING_REVIEW_TEST = OPS / "test_induction_centered_fixed_geometry_rung592_streaming_preexecution_review.py"
PREREG_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_PREREGISTRATION_REVIEW.md"
AMENDMENT_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_AMENDMENT_INDEPENDENT_REVIEW.md"
DIAGNOSTIC_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_DIAGNOSTIC_PREFIX_AMENDMENT_INDEPENDENT_REVIEW.md"
MASK_REVIEW = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_NONFINITE_MASK_AMENDMENT_INDEPENDENT_REVIEW.md"
INSTRUMENT_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG593_INSTRUMENT_REPAIR_AMENDMENT.md"
R592_POST_AUDIT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_POSTEXECUTION_AUDIT.md"
R592_POST_AUDITOR = OPS / "audit_induction_centered_fixed_geometry_rung592_postexecution.py"
BUILDER_HANDOFF = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG593_BUILDER_HANDOFF.md"
DISPATCH_AMENDMENT = POLY / "INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG593_ARG_MAX_DISPATCH_AMENDMENT.md"

MFD_ALLOW_SEALING = int(getattr(os, "MFD_ALLOW_SEALING", 0x0002))
F_ADD_SEALS = int(getattr(fcntl, "F_ADD_SEALS", 1033))
F_GET_SEALS = int(getattr(fcntl, "F_GET_SEALS", 1034))
F_SEAL_SEAL = int(getattr(fcntl, "F_SEAL_SEAL", 0x0001))
F_SEAL_SHRINK = int(getattr(fcntl, "F_SEAL_SHRINK", 0x0002))
F_SEAL_GROW = int(getattr(fcntl, "F_SEAL_GROW", 0x0004))
F_SEAL_WRITE = int(getattr(fcntl, "F_SEAL_WRITE", 0x0008))
FULL_SEAL_MASK = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL
_OS_MEMFD_DEFAULT = object()

FROZEN_HASHES = {
    PRODUCER: "193013a0c0cf1bec19be4843dee751c355d56f69fbf2d761df57baaa86c6024a",
    OWNER_TEST: "7c573951d8631e1870e6b7d565294223d15739e96d2c44dd24b3a52c840b9a43",
    FAKE_RUNTIME_TEST: "c8b7422d4cf6a3735cb0298489b648b00c2c64a32aae7b2ecf59706a32973860",
    ADAPTER_TEST: "83885a79e11d962ba2fcc0fc61e2e2ae984a4bd1643b5738bae2092470c15bae",
    RUNTIME: "768c0ed002f107c7549070a0c162552a0e1825ed3de411ff85987a79a8165777",
    DRYRUN: "a763b8f48541d152c302cd6d31127aa108f1a90abf54e07cc77ff77c224c36a1",
    PREREG: "870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a",
    AMENDMENT: "5e9fe2bcf41b88c199b5dfab2ba3ec7d0fa8f4b4b2952173c1984391e4d53094",
    DIAGNOSTIC_AMENDMENT: "f153fa3df6d7d00e951d2e7d2f0a270e6383f9133d0d34049a9eee57640b2c62",
    MASK_AMENDMENT: "f93ce1e524e6a0298a0b28f036ac35c75621c5bc80cf4cc0cac7bbe7589a99dc",
    TOPOLOGY_AMENDMENT: "15219749dd1d696e52c3129052cadce6758b7186390303eace216d98c953188e",
    TOPOLOGY_REVIEW: "7b127fc100192d2ed0eb432ad2cfbf506d151314b1e9419d1e3fa424eb487772",
    TOPOLOGY_REVIEW_TEST: "9b0ac1fe5347824135612cf675676d61d3d5f55c7b12c9d89652e0c30e7ed183",
    IMPLEMENTATION_BLOCK_REVIEW: "9b8e4ce54d1b34d650ef088f841672cf01a4482257446b611ba37e1353a457cf",
    IMPLEMENTATION_BLOCK_TEST: "3f8a559a14015498d375ba75271cf57647b9cc9841ef32b1e9e32406abf71323",
    STORAGE_AMENDMENT: "2df290b9670adfb8541d675e51fc607f856f7f70c083248fdba14ab8cf90df07",
    STORAGE_BLOCK_REVIEW: "e88ea815b154d922df44143d549c735068d6947e729d668b4849cfbd23e4f444",
    STORAGE_BLOCK_TEST: "ec1759555f8abf80cde08a93fe01c9e97fe32b6effc467085c75d06a551c6899",
    CAPACITY_AMENDMENT: "da634dd10da654739d761a6c8f8ce9c1434d8946a7477ba6d9c005c873386458",
    STREAMING_REVIEW: "8a22980fb766b8b51cac81acb69ad8e84cd886dae053613591acabc415c6f225",
    STREAMING_REVIEW_TEST: "7c84f858625b92af4b7242b168cf7d321d8dcc7ae82a5988bfcb9372d099514b",
    PREREG_REVIEW: "9b76b91995374697b8a828ce042e59d81bfddcbaa5f6e843cb0f32f6b01e57f7",
    AMENDMENT_REVIEW: "21bdc310b4798d3ae6d47fc2ed7dfee969afd871bc90db381db634e2c4cae2f5",
    DIAGNOSTIC_REVIEW: "e7373c2249e0456327d386559d4f3fa68e0661ed076a35fb120ad9d8effaa675",
    MASK_REVIEW: "b1990a81565cdd63e283ba8896cd9a57b7e8ab81064435a90ea9304d1a5a6c60",
    INSTRUMENT_AMENDMENT: "df0ceebf57818534a9b4ac5de4cd82ca64f2c1228cdfd476e350e62e5707729c",
    R592_POST_AUDIT: "1398d4907d868ff3053c3e0690861a8c0be48f19b1b1286cbbb2534d56622b46",
    R592_POST_AUDITOR: "cc36365e6dc95d6975b181ff96ad6c1f1bc44980d05c1afb25e84ff1252ddace",
    BUILDER_HANDOFF: "1cef804ca15fce531e5185ba0012d1ed3110058f7fa5a541d0d4bd16dc9c87de",
    DISPATCH_AMENDMENT: "46bf7c8821fc5988b68a2730eec59e6410a2c730d3364f5a833899edadc1a4df",
}

MINIMUM_FREE_BYTES = 9_455_639_040
SELECT_MINIMUM_FREE_BYTES = 3_954_175_488

REGISTERED_PREDICTIONS = {
    "pred_a_selector_transfer": "score and joint transfer selector changes while payload remains selective",
    "pred_b_payload_transfer": "payload and joint transfer content changes while score remains selective",
    "pred_c_active_control_selectivity": "active controls reject broad full-vocabulary damage",
}

OUTCOME_NAMESPACES = (
    ROOT / "induction_centered_fixed_geometry_rung593_results.json",
    ROOT / "induction_centered_fixed_geometry_rung593_receipt.json",
    ROOT / "induction_centered_fixed_geometry_rung593_evidence",
    ROOT / "induction_centered_fixed_geometry_rung593_invalid_diagnostic.json",
    ROOT / "induction_centered_fixed_geometry_rung593_invalid_receipt.json",
    ROOT / "induction_centered_fixed_geometry_rung593_invalid_evidence",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_bytes(bindings: Mapping[Path, str] = FROZEN_HASHES) -> dict[str, str]:
    observed = {}
    for path, expected in bindings.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen R593 file changed or missing: {path}")
        observed[str(path)] = expected
    return observed


def require_unused_namespaces(paths: Sequence[Path] = OUTCOME_NAMESPACES) -> None:
    occupied = [str(path) for path in paths if path.exists()]
    if occupied:
        raise RuntimeError(f"R593 outcome namespace already exists: {occupied}")


def require_free_space(
    path: Path, *, minimum: int = MINIMUM_FREE_BYTES, statvfs_function=os.statvfs,
) -> dict[str, int]:
    statistics = statvfs_function(path)
    observed = int(statistics.f_bavail) * int(statistics.f_frsize)
    if observed < minimum:
        raise RuntimeError(
            f"R593 insufficient free space before model boundary: {observed} < {minimum}"
        )
    return {"boundary": "model", "available_bytes": observed, "required_free_bytes": int(minimum)}


def load_frozen_producer():
    os.environ["BQLIB_NO_MODEL"] = "1"
    source = PRODUCER.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_HASHES[PRODUCER]:
        raise RuntimeError("R593 producer changed before immutable import")
    name = "r593_hash_pinned_managed_producer"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(PRODUCER))
    if spec is None:
        raise RuntimeError("cannot construct frozen R593 producer")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(PRODUCER)
    sys.modules[name] = module
    exec(compile(source, str(PRODUCER), "exec"), module.__dict__)
    return module


def run_model_free_validation() -> dict[str, object]:
    producer = load_frozen_producer()
    observed = producer.build_dryrun()
    committed = json.loads(DRYRUN.read_text(encoding="utf-8"))
    if observed != committed:
        raise RuntimeError("R593 dry run differs from frozen artifact")
    if observed["model_forwards"] != 0 or observed["model_backwards"] != 0 or observed["model_weights_updated"] is not False:
        raise RuntimeError("R593 dry run reports model work")
    if any(observed[key] for key in ("select_opened", "final_opened", "ood_opened")):
        raise RuntimeError("R593 dry run opened a sealed split")
    return observed


def preflight(
    *, namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES, capacity_path: Path = ROOT,
    statvfs_function=os.statvfs,
) -> dict[str, object]:
    observed = verify_frozen_bytes()
    require_unused_namespaces(namespace_paths)
    dryrun = run_model_free_validation()
    capacity = require_free_space(capacity_path, statvfs_function=statvfs_function)
    return {
        "schema": "execute_induction_centered_fixed_geometry_rung593_preflight_v1",
        "status": "prospective_candidate_different_agent_exact_review_required",
        "frozen_sha256": observed,
        "fit_call_manifest_sha256": dryrun["fit_call_manifest_sha256"],
        "select_call_manifest_sha256": dryrun["select_call_manifest_sha256"],
        "registered_fit_forwards": 639,
        "registered_select_forwards": 322,
        "registered_max_forwards": 961,
        "capacity_preflight": capacity,
        "capacity_thresholds": {
            "before_model": MINIMUM_FREE_BYTES,
            "before_select_after_fit": SELECT_MINIMUM_FREE_BYTES,
            "fit_canonical_data_bytes": 5_501_463_552,
            "remaining_select_plus_chunk_bytes": 2_794_172_416,
            "safety_margin_bytes": 1_160_003_072,
        },
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "select_opened": False,
        "final_opened": False,
        "ood_opened": False,
    }


def linux_memfd_create(
    name: str, flags: int = MFD_ALLOW_SEALING, *, os_function=_OS_MEMFD_DEFAULT,
) -> int:
    """Call the same Linux memfd syscall when this Python omits its os wrapper."""
    function = getattr(os, "memfd_create", None) if os_function is _OS_MEMFD_DEFAULT else os_function
    if function is not None:
        return int(function(name, flags))
    library = ctypes.CDLL(None, use_errno=True)
    try:
        fallback = library.memfd_create
    except AttributeError as symbol_exception:
        raise RuntimeError("glibc memfd_create is unavailable") from symbol_exception
    fallback.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    fallback.restype = ctypes.c_int
    descriptor = int(fallback(name.encode("ascii"), flags))
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return descriptor


def create_sealed_source(source: bytes, *, memfd_function=linux_memfd_create) -> int:
    descriptor = memfd_function("r593-immutable-producer", MFD_ALLOW_SEALING)
    try:
        view = memoryview(source)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise RuntimeError("R593 memfd source write made no progress")
            written += count
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, F_ADD_SEALS, FULL_SEAL_MASK)
        observed_seals = int(fcntl.fcntl(descriptor, F_GET_SEALS))
        if observed_seals != FULL_SEAL_MASK:
            raise RuntimeError(
                f"R593 memfd seal mask changed: {observed_seals} != {FULL_SEAL_MASK}"
            )
        os.set_inheritable(descriptor, True)
        if not os.get_inheritable(descriptor):
            raise RuntimeError("R593 memfd descriptor is not inheritable")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def sealed_python_command(
    source: bytes, *, logical_path: str, expected_length: int,
    expected_sha256: str, adapter_sha256: str,
) -> tuple[str, list[str], int]:
    descriptor = create_sealed_source(source)
    launcher = f"""import hashlib,os,sys
f=int(sys.argv[1]); n={int(expected_length)!r}; h={expected_sha256!r}
try:
 b=b''.join(iter(lambda:os.read(f,65536),b''))
finally:
 os.close(f)
if len(b)!=n or hashlib.sha256(b).hexdigest()!=h:
 raise RuntimeError('R593 sealed producer length/hash mismatch')
p={logical_path!r}; sys.argv=[p]
exec(compile(b,p,'exec'),{{'__name__':'__main__','__file__':p,'__package__':None,'__r593_immutable_sha256__':h,'__r593_adapter_sha256__':{adapter_sha256!r}}})
"""
    argv = [sys.executable, "-I", "-c", launcher, str(descriptor)]
    if any(len(argument.encode("utf-8")) >= 4096 for argument in argv):
        os.close(descriptor)
        raise RuntimeError("R593 sealed launcher exceeded 4095 bytes per argument")
    return sys.executable, argv, descriptor


def scientific_command() -> tuple[str, list[str], int]:
    """Dispatch verified bytes through an anonymous, fully sealed memory file."""
    source = PRODUCER.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_HASHES[PRODUCER]:
        raise RuntimeError("R593 producer changed before immutable dispatch")
    return sealed_python_command(
        source, logical_path=str(PRODUCER), expected_length=len(source),
        expected_sha256=FROZEN_HASHES[PRODUCER],
        adapter_sha256=hashlib.sha256(ADAPTER.read_bytes()).hexdigest(),
    )


def dispatch(environment: Mapping[str, str], *, exec_function=os.execv,
             namespace_paths: Sequence[Path] = OUTCOME_NAMESPACES,
             capacity_path: Path = ROOT, statvfs_function=os.statvfs) -> dict[str, object]:
    plan = preflight(
        namespace_paths=namespace_paths, capacity_path=capacity_path,
        statvfs_function=statvfs_function,
    )
    mode = environment.get("BQLIB_DRYRUN")
    if mode == "1":
        plan["mode"] = "model_free_dryrun"
        return plan
    if mode is not None:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")
    executable, argv, descriptor = scientific_command()
    try:
        exec_function(executable, argv)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    raise RuntimeError("R593 scientific os.execv unexpectedly returned")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("R593 preflight adapter accepts no command-line arguments")
    report = dispatch(os.environ)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
