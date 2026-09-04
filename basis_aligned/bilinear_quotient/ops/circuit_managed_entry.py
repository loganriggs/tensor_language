"""Hash-first immutable managed entry for compiled circuit experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Mapping, Sequence

from circuit_experiment_spec import ArtifactRef, CircuitExperimentSpec, validate_spec


class ManagedEntryError(RuntimeError):
    """Managed preflight cannot prove an immutable outcome-blind execution."""


@dataclass(frozen=True)
class ModuleBinding:
    role: str
    module_name: str
    is_package: bool = False
def _safe_open_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ManagedEntryError(f"cannot safely open frozen artifact: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManagedEntryError(f"frozen artifact is not a regular file: {path}")
        chunks: list[bytes] = []
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
        raise ManagedEntryError(f"frozen artifact changed during capture: {path}")
    return b"".join(chunks)
def capture_frozen_artifacts(
    spec: CircuitExperimentSpec, *, base_dir: Path, dryrun: bool = False
) -> dict[str, bytes]:
    """Capture declared bytes, excluding forbidden outcome bytes in dry-run mode."""
    validate_spec(spec)
    captured: dict[str, bytes] = {}
    root = base_dir.resolve()
    for reference in spec.artifacts:
        if dryrun and not reference.dryrun_access:
            continue
        relative = Path(reference.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ManagedEntryError(f"artifact path is not relative and contained: {reference.path}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ManagedEntryError(f"artifact escaped base directory: {reference.path}")
        data = _safe_open_bytes(path)
        observed = hashlib.sha256(data).hexdigest()
        if observed != reference.sha256:
            raise ManagedEntryError(
                f"frozen artifact changed: {reference.role}; "
                f"expected={reference.sha256}, observed={observed}"
            )
        captured[reference.role] = data
    return captured
def validate_dryrun_closure(spec: CircuitExperimentSpec) -> None:
    """Forbid all outcome-bearing bytes from the advertised model-free path."""
    for reference in spec.artifacts:
        if reference.kind not in {"source", "prereg", "authority", "outcome"}:
            raise ManagedEntryError("artifact kind must be typed")
        if reference.kind == "outcome" and reference.dryrun_access:
            raise ManagedEntryError(f"dry run reaches outcome artifact: {reference.role}")


def module_from_verified_bytes(
    name: str, path_label: str, source: bytes, *, is_package: bool = False
) -> ModuleType:
    """Compile exactly the captured bytes; never reopen the verified path."""
    module = ModuleType(name)
    module.__file__ = path_label
    module.__package__ = name if is_package else name.rpartition(".")[0]
    if is_package:
        module.__path__ = [str(Path(path_label).parent)]
    sys.modules[name] = module
    try:
        exec(compile(source, path_label, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is None:
            sys.modules.pop(name, None)
            raise ManagedEntryError(f"module parent was not loaded first: {parent_name}")
        setattr(parent, child_name, module)
    return module


def load_verified_modules(
    spec: CircuitExperimentSpec,
    captured: Mapping[str, bytes],
    bindings: Sequence[ModuleBinding],
) -> dict[str, ModuleType]:
    references = {reference.role: reference for reference in spec.artifacts}
    if len({binding.role for binding in bindings}) != len(bindings) \
            or len({binding.module_name for binding in bindings}) != len(bindings):
        raise ManagedEntryError("module bindings are duplicated")
    modules: dict[str, ModuleType] = {}
    for binding in bindings:
        reference = references.get(binding.role)
        if reference is None or not reference.executable or reference.kind != "source":
            raise ManagedEntryError(f"module binding is not an executable source: {binding.role}")
        if binding.role not in captured:
            raise ManagedEntryError(f"verified source bytes are absent: {binding.role}")
        modules[binding.role] = module_from_verified_bytes(
            binding.module_name, reference.path, captured[binding.role],
            is_package=binding.is_package,
        )
    declared_executables = {
        reference.role for reference in spec.artifacts if reference.executable
    }
    if set(modules) != declared_executables:
        raise ManagedEntryError(
            f"executable closure mismatch: missing={sorted(declared_executables - set(modules))}"
        )
    return modules


def dispatch(
    spec: CircuitExperimentSpec,
    *,
    base_dir: Path,
    bindings: Sequence[ModuleBinding],
    producer_role: str,
    environment: Mapping[str, str],
    dryrun_function: str = "run_dryrun",
    science_function: str = "run_science",
) -> object:
    """Run the dry or real branch on one pre-import immutable snapshot."""
    mode = environment.get("BQLIB_DRYRUN")
    if mode not in (None, "1"):
        raise ManagedEntryError("BQLIB_DRYRUN must be absent or exactly '1'")
    validate_dryrun_closure(spec)
    captured = capture_frozen_artifacts(spec, base_dir=base_dir, dryrun=mode == "1")
    modules = load_verified_modules(spec, captured, bindings)
    if producer_role not in modules:
        raise ManagedEntryError("producer role is outside executable closure")
    producer = modules[producer_role]
    if mode == "1":
        function = getattr(producer, dryrun_function, None)
    else:
        function = getattr(producer, science_function, None)
    if not callable(function):
        raise ManagedEntryError("managed producer entry function is absent")
    return function()
