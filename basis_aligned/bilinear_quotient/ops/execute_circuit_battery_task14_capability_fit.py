#!/usr/bin/env python3
# BQGATE: EXPERIMENT
"""Hash-bound, execution-blocked managed adapter for task14 capability FIT."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping, Sequence


REPO_ROOT = Path("/workspace/tensor_language")
ADAPTER = Path(__file__).resolve()
COMPILER_COMMIT = "fc586c1158ddeee7df8f4b502deec54189609c4c"
COMPILER_REVIEW_COMMIT = "10afc5d6005d169879b07e92cb5fcb4e3a65f312"
EXECUTION_AUTHORIZED = False

REGISTERED_PREDICTIONS = {
    "pred_a_exact_instrument": (
        "eight ordered calls and 256 FIT row-sides expose only answer/foil float32 arrays"
    ),
    "pred_b_native_capability": "the exact frozen subject-verb agreement gates pass",
    "pred_c_opposing_capability_fail": "the exact complement is an all-null hard abort",
}


@dataclass(frozen=True)
class FrozenFile:
    role: str
    relative_path: str
    sha256: str
    kind: str
    module_name: str | None = None
    is_package: bool = False
    dryrun_access: bool = True


FILES = (
    FrozenFile(
        "result_contract", "basis_aligned/bilinear_quotient/ops/result_contract.py",
        "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272",
        "source", "result_contract",
    ),
    FrozenFile(
        "experiment_spec", "basis_aligned/bilinear_quotient/ops/circuit_experiment_spec.py",
        "64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c",
        "source", "circuit_experiment_spec",
    ),
    FrozenFile(
        "artifact_package", "basis_aligned/bilinear_quotient/ops/circuit_artifact_package.py",
        "6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc",
        "source", "circuit_artifact_package",
    ),
    FrozenFile(
        "battery_contract",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_integration_contract.py",
        "b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e",
        "source", "circuit_battery_integration_contract",
    ),
    FrozenFile(
        "managed_entry", "basis_aligned/bilinear_quotient/ops/circuit_managed_entry.py",
        "1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81",
        "source", "circuit_managed_entry",
    ),
    FrozenFile(
        "task14_generator", "basis_aligned/bilinear_quotient/ops/circuit_battery_task14.py",
        "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94",
        "source", "circuit_battery_task14",
    ),
    FrozenFile(
        "capability_compiler",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_capability_fit.py",
        "98b2d263c5120c1a7b700dc4bb451f65cc9f9b338740d2cfbc7ae25a3ba5aab1",
        "source", "circuit_battery_task14_capability_fit",
    ),
    FrozenFile(
        "producer",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_capability_fit_producer.py",
        "9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e",
        "source", "circuit_battery_task14_capability_fit_producer",
    ),
    FrozenFile(
        "capability_preregistration",
        "basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK14_SUBJECT_VERB_AGREEMENT_CAPABILITY_FIT_PREREGISTRATION.md",
        "06a9747b4707999e11637a45cf83588bfd9cb8671d6b3a25790518af62900f8b",
        "prereg",
    ),
    FrozenFile(
        "producer_implementation_preregistration",
        "basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK14_CAPABILITY_FIT_PRODUCER_IMPLEMENTATION_PREREGISTRATION_2026-09-04.md",
        "d84d345c8d2b4183979cd09a57d60c87fccc5a36f03bddf0fa9316f07779a6f3",
        "prereg",
    ),
    FrozenFile(
        "compiler_review",
        "basis_aligned/polynomial_causal/TASK14_SUBJECT_VERB_AGREEMENT_CAPABILITY_FIT_COMPILER_REVIEW_2026-09-04.md",
        "a1707dd88949a9b5beb439b275e665cda1a7a62a6d5eedf076d20d192c852e59",
        "prereg",
    ),
    FrozenFile(
        "fit_authority",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_agreement_fit_authority.json",
        "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f",
        "authority",
    ),
    FrozenFile(
        "receipt_source", "basis_aligned/bilinear_quotient/ops/receipt.py",
        "ced8065d262d3ae8b1ac958424848ccf75d4264174b0f4b1b3144d0f4be99708",
        "source", "receipt", False, False,
    ),
    FrozenFile(
        "jacclust_package", "jacclust/__init__.py",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "source", "jacclust", True, False,
    ),
    FrozenFile(
        "model_source", "jacclust/tt_model.py",
        "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
        "source", "jacclust.tt_model", False, False,
    ),
    FrozenFile(
        "observed_model_facade", "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
        "source", "bilin18_observed_model_facade", False, False,
    ),
    FrozenFile(
        "fastload_dependency",
        "basis_aligned/bilinear_quotient/ops/mlp_in_situ_usage_rank_map_probe.py",
        "c701af71491d29f33f5ad691f89380a9fa7c2d86514a61fd7423ad8a78fd4d16",
        "source", "mlp_in_situ_usage_rank_map_probe", False, False,
    ),
    FrozenFile(
        "fastload_source", "basis_aligned/bilinear_quotient/ops/fastload.py",
        "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
        "source", "fastload", False, False,
    ),
    FrozenFile(
        "canary1_source", "basis_aligned/bilinear_quotient/bilin18_canary.py",
        "3316a60e18d518f4c619d69b95ec4db34e1c72ad159f6bc4842405231b6a84f8",
        "source", None, False, False,
    ),
    FrozenFile(
        "canary2_source", "basis_aligned/bilinear_quotient/bilin18_canary2.py",
        "cc092508a9d7eee357cbe87d10c226357fcc3257ca6c456efa4a8054b4bf5a23",
        "source", None, False, False,
    ),
)

BOOTSTRAP_ROLES = ("result_contract", "experiment_spec", "managed_entry")
BASE_LOAD_ORDER = (
    "result_contract", "experiment_spec", "artifact_package", "battery_contract",
    "managed_entry", "task14_generator", "capability_compiler", "producer",
)
REAL_LOAD_ORDER = (
    "receipt_source", "jacclust_package", "model_source", "observed_model_facade",
    "fastload_dependency", "fastload_source",
)


class AdapterError(RuntimeError):
    """Frozen closure or execution boundary changed."""


def safe_read(path: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AdapterError(f"cannot safely open frozen file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterError(f"frozen path is not regular: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    )
    if identity(before) != identity(after):
        raise AdapterError(f"frozen file changed during capture: {path}")
    payload = b"".join(chunks)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise AdapterError(
            f"frozen file changed: {path}; expected={expected_sha256}, observed={observed}"
        )
    return payload


def load_module(name: str, path: str, source: bytes, *, is_package: bool = False) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = path
    module.__package__ = name if is_package else name.rpartition(".")[0]
    if is_package:
        module.__path__ = [str(Path(path).parent)]
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, path, "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is None:
            raise AdapterError(f"verified module parent absent: {parent_name}")
        setattr(parent, child_name, module)
    return module


def file_by_role(role: str) -> FrozenFile:
    matches = [item for item in FILES if item.role == role]
    if len(matches) != 1:
        raise AdapterError(f"frozen role missing or duplicated: {role}")
    return matches[0]


def bootstrap() -> tuple[ModuleType, ModuleType]:
    loaded = {}
    for role in BOOTSTRAP_ROLES:
        item = file_by_role(role)
        loaded[role] = load_module(
            str(item.module_name), item.relative_path,
            safe_read(REPO_ROOT / item.relative_path, item.sha256),
            is_package=item.is_package,
        )
    return loaded["experiment_spec"], loaded["managed_entry"]


def execution_spec(compiler: ModuleType):
    artifacts = tuple(
        compiler.ArtifactRef(
            role=item.role, path=item.relative_path, sha256=item.sha256, kind=item.kind,
            executable=item.module_name is not None, dryrun_access=item.dryrun_access,
        )
        for item in FILES
    )
    spec = compiler.CircuitExperimentSpec(
        experiment_id="circuit-battery-task14-capability-fit-execution-blocked-v1",
        rung=14,
        artifacts=artifacts,
        phases=(compiler.PhaseSpec(
            "FIT", opens_after=None, forbidden_splits=("SELECT", "TEST", "OOD")
        ),),
        authority_tables=(), calls=(),
    )
    compiler.validate_spec(spec)
    return spec


def validate_captured_bytes(captured: Mapping[str, bytes]) -> None:
    forbidden_roles = {
        role for role in captured
        if "authorization" in role.lower() or "producer_review" in role.lower()
    }
    if forbidden_roles:
        raise AdapterError("blocked adapter captured an authorization or producer review")
    for role, payload in captured.items():
        item = file_by_role(role)
        if type(payload) is not bytes or hashlib.sha256(payload).hexdigest() != item.sha256:
            raise AdapterError(f"captured frozen bytes changed before module load: {role}")


def capture(mode: str) -> tuple[ModuleType, ModuleType, dict[str, bytes]]:
    compiler, managed = bootstrap()
    spec = execution_spec(compiler)
    managed.validate_dryrun_closure(spec)
    captured = managed.capture_frozen_artifacts(spec, base_dir=REPO_ROOT, dryrun=True)
    expected = {item.role for item in FILES if item.dryrun_access}
    if mode != "1" or set(captured) != expected:
        raise AdapterError("managed closure differs from blocked dryrun mode")
    validate_captured_bytes(captured)
    return compiler, managed, captured


def load_verified_closure(
    managed: ModuleType, captured: Mapping[str, bytes], *, real: bool,
) -> dict[str, ModuleType]:
    if real:
        raise AdapterError("real closure loading is disabled in this adapter")
    validate_captured_bytes(captured)
    loaded = {}
    for role in BASE_LOAD_ORDER:
        item = file_by_role(role)
        payload = captured.get(role)
        if payload is None:
            raise AdapterError(f"verified executable bytes absent: {role}")
        loaded[role] = managed.module_from_verified_bytes(
            str(item.module_name), item.relative_path, payload, is_package=item.is_package
        )
    if loaded["producer"].capability is not loaded["capability_compiler"] \
            or loaded["producer"].package is not loaded["artifact_package"] \
            or loaded["producer"].framework is not loaded["experiment_spec"]:
        raise AdapterError("producer imports did not resolve to captured modules")
    return loaded


def dispatch(environment: Mapping[str, str]) -> dict[str, object]:
    mode = environment.get("BQLIB_DRYRUN")
    if mode not in (None, "1"):
        raise AdapterError("BQLIB_DRYRUN must be absent or exactly '1'")
    if mode is None:
        raise AdapterError(
            "task14 capability execution is unauthorized: producer review and prospective authorization are absent"
        )
    _, managed, captured = capture(mode)
    producer = load_verified_closure(managed, captured, real=False)["producer"]
    report = producer.run_dryrun(captured)
    report["execution_authorized"] = EXECUTION_AUTHORIZED
    report["status"] = "model_free_plan_validated_execution_unauthorized"
    report["adapter_sha256"] = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
    report["captured_roles"] = sorted(captured)
    report["runtime_only_roles_excluded"] = sorted(
        item.role for item in FILES if not item.dryrun_access
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("task14 capability managed adapter accepts no arguments")
    print(json.dumps(dispatch(os.environ), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
