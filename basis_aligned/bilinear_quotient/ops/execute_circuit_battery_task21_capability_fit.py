#!/usr/bin/env python3
"""Prospectively authorized adapter successor for task-21 FIT native capability.

The model-free branch validates the exact reviewed compiler and producer plan.
The real branch is authorized only by an exact captured amendment and producer
review, and remains ineligible for enqueue until this successor is independently
reviewed at its new immutable hash.
"""

# BQGATE: EXPERIMENT

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
COMPILER_COMMIT = "9ebab94615eade27b1eb63e4f2c6239337b71dc9"
COMPILER_REVIEW_COMMIT = "ca088ce0906160958a2586cff50b707699b7eb88"
PRODUCER_BUILD_COMMIT = "000a113eed35c7e8fac0d2ceed126925963cd0d7"
PRODUCER_REVIEW_COMMIT = "6b8fe576594bb82a5a2093f2338603040739c9af"
EXECUTION_AUTHORIZED = True

REGISTERED_PREDICTIONS = {
    "pred_a_exact_instrument": (
        "all eight exact calls and 168 row-side evaluations expose only the two registered arrays"
    ),
    "pred_b_native_capability": (
        "both side accuracies are at least .90, every transform cell is at least .85, and mean margins are positive"
    ),
    "pred_c_opposing_capability_fail": (
        "the exact logical complement is an all-null hard-abort terminal"
    ),
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
        "battery_contract", "basis_aligned/bilinear_quotient/ops/circuit_battery_integration_contract.py",
        "b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e",
        "source", "circuit_battery_integration_contract",
    ),
    FrozenFile(
        "managed_entry", "basis_aligned/bilinear_quotient/ops/circuit_managed_entry.py",
        "1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81",
        "source", "circuit_managed_entry",
    ),
    FrozenFile(
        "task21_adapter", "basis_aligned/bilinear_quotient/ops/circuit_battery_task21.py",
        "bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2",
        "source", "circuit_battery_task21",
    ),
    FrozenFile(
        "capability_compiler",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task21_capability_fit.py",
        "43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390",
        "source", "circuit_battery_task21_capability_fit",
    ),
    FrozenFile(
        "producer",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task21_capability_fit_producer.py",
        "395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf",
        "source", "circuit_battery_task21_capability_fit_producer",
    ),
    FrozenFile(
        "capability_preregistration",
        "basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK21_VERBATIM_COPY_CAPABILITY_FIT_PREREGISTRATION.md",
        "da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3",
        "prereg",
    ),
    FrozenFile(
        "producer_implementation_preregistration",
        "basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK21_CAPABILITY_FIT_PRODUCER_IMPLEMENTATION_PREREGISTRATION_2026-09-04.md",
        "3009aff99543e34e8a7d33a486035e5168c136f168c18ebb8e3fd8a3ad290882",
        "prereg",
    ),
    FrozenFile(
        "compiler_review",
        "basis_aligned/polynomial_causal/TASK21_VERBATIM_COPY_AUTHORITY_COMPILER_REPRODUCIBILITY_REVIEW_2026-09-04.md",
        "3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50",
        "prereg",
    ),
    FrozenFile(
        "producer_review",
        "basis_aligned/polynomial_causal/TASK21_CAPABILITY_FIT_PRODUCER_BLOCKED_ADAPTER_REVIEW_2026-09-04.md",
        "8763602a753345a19312613160d32b3ffe537a7ebfcb4bcf4c83905a25b7ed29",
        "prereg",
    ),
    FrozenFile(
        "authorization_amendment",
        "basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK21_CAPABILITY_FIT_AUTHORIZATION_AMENDMENT_2026-09-04.md",
        "a31cf24ec79d86f084c29bdc18a909e1ff0457b4e0921fd6249f722adf2b08d1",
        "prereg",
    ),
    FrozenFile(
        "fit_authority",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_fit_authority.json",
        "69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94",
        "authority",
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
    "managed_entry", "task21_adapter", "capability_compiler", "producer",
)
REAL_LOAD_ORDER = ("jacclust_package", "model_source", "observed_model_facade")
AUTHORIZATION_ROLES = ("producer_review", "authorization_amendment")


class AdapterError(RuntimeError):
    """The immutable execution closure or authorization boundary changed."""


def safe_read(path: Path, expected_sha256: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AdapterError(f"cannot safely open frozen file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterError(f"frozen path is not a regular file: {path}")
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


def load_module(name: str, path_label: str, source: bytes, *, is_package: bool = False) -> ModuleType:
    """Replace any import-cache entry with exactly the captured source bytes."""
    module = ModuleType(name)
    module.__file__ = path_label
    module.__package__ = name if is_package else name.rpartition(".")[0]
    if is_package:
        module.__path__ = [str(Path(path_label).parent)]
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, path_label, "exec"), module.__dict__)
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
            raise AdapterError(f"verified module parent was not loaded: {parent_name}")
        setattr(parent, child_name, module)
    return module


def file_by_role(role: str) -> FrozenFile:
    matches = [item for item in FILES if item.role == role]
    if len(matches) != 1:
        raise AdapterError(f"frozen role is missing or duplicated: {role}")
    return matches[0]


def bootstrap() -> tuple[ModuleType, ModuleType]:
    loaded = {}
    for role in BOOTSTRAP_ROLES:
        item = file_by_role(role)
        payload = safe_read(REPO_ROOT / item.relative_path, item.sha256)
        loaded[role] = load_module(
            str(item.module_name), item.relative_path, payload, is_package=item.is_package
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
        experiment_id="circuit-battery-task21-capability-fit-execution-v1",
        rung=21,
        artifacts=artifacts,
        phases=(compiler.PhaseSpec(
            "FIT", opens_after=None, forbidden_splits=("SELECT", "TEST", "OOD")
        ),),
        authority_tables=(),
        calls=(),
    )
    compiler.validate_spec(spec)
    return spec


def capture(mode: str | None) -> tuple[ModuleType, ModuleType, dict[str, bytes]]:
    compiler, managed = bootstrap()
    spec = execution_spec(compiler)
    managed.validate_dryrun_closure(spec)
    captured = managed.capture_frozen_artifacts(spec, base_dir=REPO_ROOT, dryrun=mode == "1")
    expected = {item.role for item in FILES if mode != "1" or item.dryrun_access}
    if set(captured) != expected:
        raise AdapterError("managed artifact closure differs from the execution mode")
    validate_captured_bytes(captured)
    return compiler, managed, captured


def validate_captured_bytes(captured: Mapping[str, bytes]) -> None:
    for role, payload in captured.items():
        item = file_by_role(role)
        if type(payload) is not bytes or hashlib.sha256(payload).hexdigest() != item.sha256:
            raise AdapterError(f"captured frozen bytes changed before module load: {role}")
    if not set(AUTHORIZATION_ROLES).issubset(captured):
        raise AdapterError("exact authorization amendment and producer review were not captured")


def load_verified_closure(
    managed: ModuleType, captured: Mapping[str, bytes], *, real: bool,
) -> dict[str, ModuleType]:
    validate_captured_bytes(captured)
    loaded = {}
    order = BASE_LOAD_ORDER + (REAL_LOAD_ORDER if real else ())
    for role in order:
        item = file_by_role(role)
        payload = captured.get(role)
        if payload is None:
            raise AdapterError(f"verified executable bytes are absent: {role}")
        loaded[role] = managed.module_from_verified_bytes(
            str(item.module_name), item.relative_path, payload, is_package=item.is_package
        )
    if loaded["producer"].capability is not loaded["capability_compiler"] \
            or loaded["producer"].package is not loaded["artifact_package"] \
            or loaded["producer"].framework is not loaded["experiment_spec"]:
        raise AdapterError("producer imports did not resolve to verified captured modules")
    return loaded


def dispatch(environment: Mapping[str, str]) -> dict[str, object]:
    mode = environment.get("BQLIB_DRYRUN")
    if mode not in (None, "1"):
        raise AdapterError("BQLIB_DRYRUN must be absent or exactly '1'")
    if mode is None and not EXECUTION_AUTHORIZED:
        raise AdapterError(
            "task21 capability execution is not authorized: fresh producer review and a prospective authorization are absent"
        )
    _, managed, captured = capture(mode)
    modules = load_verified_closure(managed, captured, real=mode is None)
    producer = modules["producer"]
    if mode == "1":
        report = producer.run_dryrun(captured)
        report["execution_authorized"] = EXECUTION_AUTHORIZED
        report["status"] = "model_free_plan_validated_authorized_adapter_pending_final_review"
        report["adapter_sha256"] = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
        report["captured_roles"] = sorted(captured)
        report["runtime_only_roles_excluded"] = sorted(
            item.role for item in FILES if not item.dryrun_access
        )
        return report
    return producer.run_science(captured)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        raise SystemExit("task21 capability managed adapter accepts no arguments")
    print(json.dumps(dispatch(os.environ), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
