"""Pre-outcome authority and immutable identity helpers for the tangent pilot."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RANK640_PARENT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
FIT_ROWS = HERE.parent / "bilinear_quotient/.rowcache/fineweb_n480_skip80.pt"
ROW_AUTHORITY = HERE.parent / "bilinear_quotient/.rowcache/fineweb_oracle_v2_receipt.json"
TANGENT_ROWS = HERE.parent / "bilinear_quotient/.rowcache/fineweb_n96_skip80.pt"
PLAN = HERE / "finite_horizon_tangent_plan.json"
PREREG = HERE / "FINITE_HORIZON_TANGENT_REALIZATION_PREREGISTRATION.md"

EXPECTED_IMMUTABLE_SHA256 = {
    str(RANK640_PARENT): "639fb8480efee790403113079333100bd63bb61426f6fd6e4dcebd89b21c337d",
    str(CAUSAL_PARENT): "73bd18ee81067775680b7d579036e6ec8c04b41116cd3e516b8460a7e7c7ab20",
    str(FIT_ROWS): "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
    str(ROW_AUTHORITY): "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16",
    str(TANGENT_ROWS): "94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda",
}
EXPECTED_TANGENT_RAW_SHA256 = "a703cadb1a5e27497cba43d21bca889a1d765b861c3da311a1dc4dfeb28b21cc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_raw_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def git_identity() -> dict[str, str]:
    def read(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    identity = {
        "head": read("rev-parse", "HEAD"),
        "origin_main": read("rev-parse", "origin/main"),
    }
    if identity["head"] != identity["origin_main"]:
        raise RuntimeError("tangent launch commit is not the pushed origin/main commit")
    return identity


def configure_production_runtime() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("production tangent lifecycle requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda),
        "device_index": device, "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def require_committed_sources(source_paths: Sequence[Path]) -> None:
    relative = [str(path.resolve().relative_to(ROOT)) for path in source_paths]
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--", *relative), cwd=ROOT,
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"tangent source closure is not committed: {status}")


def protected_snapshot(source_paths: Sequence[Path]) -> dict[str, Any]:
    """Validate every protected input before deserializing rows or loading a model."""
    require_committed_sources(source_paths)
    observed_immutable = {
        name: sha256_file(Path(name)) for name in EXPECTED_IMMUTABLE_SHA256
    }
    if observed_immutable != EXPECTED_IMMUTABLE_SHA256:
        changed = {
            name: {"expected": EXPECTED_IMMUTABLE_SHA256[name], "observed": value}
            for name, value in observed_immutable.items()
            if value != EXPECTED_IMMUTABLE_SHA256[name]
        }
        raise RuntimeError(f"protected tangent input changed: {changed}")
    rank_parent = json.loads(RANK640_PARENT.read_text())
    causal_parent = json.loads(CAUSAL_PARENT.read_text())
    if rank_parent.get("status") != "pass" or rank_parent.get("rank") != 640 or (
        rank_parent.get("provenance", {}).get("fit") != EXPECTED_IMMUTABLE_SHA256[str(FIT_ROWS)]
    ) or causal_parent.get("status") != "rank640_robust_pass" or (
        causal_parent.get("provenance", {}).get("fit")
        != EXPECTED_IMMUTABLE_SHA256[str(FIT_ROWS)]
    ):
        raise RuntimeError("admitted rank640 parent semantics changed")
    authority = json.loads(ROW_AUTHORITY.read_text())
    entry = authority.get("entries", {}).get("n96_skip80", {})
    if authority.get("authority") != "pinned_local_ordered_manifest" or not authority.get(
        "authorized_for_scored_experiments"
    ) or entry.get("tensor_raw_sha256") != EXPECTED_TANGENT_RAW_SHA256:
        raise RuntimeError("tangent row authority semantics changed")
    sources = {str(path.resolve()): sha256_file(path.resolve()) for path in source_paths}
    if len(sources) != len(tuple(source_paths)):
        raise RuntimeError("tangent source closure contains duplicate paths")
    snapshot = {
        "immutable_inputs": observed_immutable,
        "source_closure": sources,
        "plan_sha256": sha256_file(PLAN),
        "preregistration_sha256": sha256_file(PREREG),
        "git": git_identity(),
    }
    snapshot["fingerprint"] = canonical_sha256(snapshot)
    return snapshot


def validate_loaded_rows(rows: torch.Tensor) -> dict[str, Any]:
    if not torch.is_tensor(rows) or tuple(rows.shape) != (96, 513) or rows.dtype != torch.int64:
        raise RuntimeError("loaded tangent rows have the wrong shape or dtype")
    raw = tensor_raw_sha256(rows)
    if raw != EXPECTED_TANGENT_RAW_SHA256:
        raise RuntimeError("loaded tangent row bytes changed")
    return {
        "shape": list(rows.shape), "dtype": str(rows.dtype),
        "serialized_sha256": sha256_file(TANGENT_ROWS), "tensor_raw_sha256": raw,
    }


def validate_program_receipt(receipt: Mapping[str, Any]) -> None:
    parent = json.loads(RANK640_PARENT.read_text())
    if receipt.get("checkpoint") != parent.get("checkpoint"):
        raise RuntimeError("fresh program checkpoint differs from admitted rank640 parent")
    if receipt.get("attention_fit") != parent.get("fit"):
        raise RuntimeError("fresh program fit receipt differs from admitted rank640 parent")
    parent_cost = dict(parent.get("cost", {}))
    derived = {
        key: parent_cost.pop(key, None) for key in (
            "dense_reference_stored_values", "stored_values_saved",
            "stored_fraction_of_dense",
        )
    }
    dense = 545_904_054
    total = int(parent_cost.get("total_stored_values", -1))
    if derived != {
        "dense_reference_stored_values": dense,
        "stored_values_saved": dense - total,
        "stored_fraction_of_dense": total / dense,
    }:
        raise RuntimeError("admitted rank640 parent derived cost fields changed")
    if receipt.get("cost") != parent_cost:
        raise RuntimeError("fresh program cost differs from admitted rank640 parent")


def admitted_program_cost() -> dict[str, Any]:
    cost = dict(json.loads(RANK640_PARENT.read_text())["cost"])
    for key in (
        "dense_reference_stored_values", "stored_values_saved", "stored_fraction_of_dense",
    ):
        cost.pop(key)
    return cost


def program_buffer_manifest(program: torch.nn.Module, *, chunk_bytes: int = 16 << 20) -> dict[str, Any]:
    """Hash every owned program buffer in canonical name/shape/dtype/byte order."""
    if tuple(program.parameters()):
        raise RuntimeError("owned tangent program unexpectedly has parameters")
    entries: list[dict[str, Any]] = []
    tree = hashlib.sha256()
    total_values = 0
    total_bytes = 0
    for name, value in sorted(
        program.named_buffers(remove_duplicate=False), key=lambda item: item[0],
    ):
        tensor = value.detach().contiguous()
        metadata = {
            "name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "values": tensor.numel(), "bytes": tensor.numel() * tensor.element_size(),
        }
        framed = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        value_hash = hashlib.sha256()
        byte_view = tensor.reshape(-1).view(torch.uint8)
        for start in range(0, byte_view.numel(), chunk_bytes):
            block = byte_view[start:start + chunk_bytes].cpu().numpy().tobytes(order="C")
            value_hash.update(block)
        metadata["sha256"] = value_hash.hexdigest()
        tree.update(len(framed).to_bytes(8, "big"))
        tree.update(framed)
        tree.update(bytes.fromhex(metadata["sha256"]))
        entries.append(metadata)
        total_values += tensor.numel()
        total_bytes += tensor.numel() * tensor.element_size()
    result = {
        "entries": entries, "buffers": len(entries), "total_values": total_values,
        "total_bytes": total_bytes, "tree_sha256": tree.hexdigest(),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


class RunLock:
    def __init__(self, path: Path, descriptor: int, token: str) -> None:
        self.path = path
        self.descriptor = descriptor
        self.token = token
        stat = os.fstat(descriptor)
        self.device = stat.st_dev
        self.inode = stat.st_ino

    def owned(self) -> bool:
        try:
            stat = self.path.stat()
            content = self.path.read_text()
        except FileNotFoundError:
            return False
        return stat.st_dev == self.device and stat.st_ino == self.inode and (
            content == f"pid={os.getpid()} token={self.token}\n"
        )

    def assert_owned(self) -> None:
        if not self.owned():
            raise RuntimeError("tangent run lock was removed or replaced")


@contextmanager
def exclusive_run_lock(path: Path) -> Iterator[RunLock]:
    token = secrets.token_hex(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    claim = RunLock(path, descriptor, token)
    try:
        os.write(descriptor, f"pid={os.getpid()} token={token}\n".encode())
        os.fsync(descriptor)
        claim.assert_owned()
        yield claim
    finally:
        os.close(descriptor)
        if claim.owned():
            path.unlink()


def publish_json_create_only(
    path: Path, value: Mapping[str, Any], *,
    ownership_check: Callable[[], None] | None = None,
) -> None:
    """Crash-safe staged publication with atomic create-only installation."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        if ownership_check is not None:
            ownership_check()
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_torch_create_only(
    path: Path, value: Any, *, ownership_check: Callable[[], None] | None = None,
) -> None:
    """Crash-safe create-only installation of a torch tensor artifact."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if ownership_check is not None:
            ownership_check()
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
